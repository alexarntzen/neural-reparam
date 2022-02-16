import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import dataset

from deepthermal.FFNN_model import larning_rates

# penalty = 1er
# penalty_ = torch.tensor(penalty)
zero_ = torch.tensor(0.0)
l2_loss = nn.MSELoss()


@torch.no_grad()
def r_cost(
    state_index: torch.LongTensor, next_state_index: torch.LongTensor, data: dataset
) -> torch.Tensor:
    assert torch.all(next_state_index >= state_index), "wrong direction"
    if state_index[0] == next_state_index[0]:
        return zero_
    # elif torch.any(state_index == next_state_index):
    #     return penalty_

    t_data, q_data, r_data = data[:]
    q_eval = q_data[state_index[0] : next_state_index[0] + 1]
    r_eval = r_data[state_index[1] : next_state_index[1] + 1]

    index_diff = next_state_index - state_index
    if state_index[1] == next_state_index[1]:
        # if x does not change compute
        tx_indices = torch.arange(0, index_diff[0] + 1)
        integrand = torch.sum(q_eval ** 2, dim=-1)

    else:
        gcd = torch.gcd(index_diff[..., 0], index_diff[..., 1])

        # find the lowest iteger to represent the points on the interval using ints
        product_index = torch.div(
            torch.prod(index_diff, dtype=torch.long), gcd, rounding_mode="floor"
        )
        # one extra since we want the end state included
        common_length = product_index + 1

        # compute common indexes
        t_spacing = torch.div(index_diff[1], gcd, rounding_mode="floor").item()
        x_spacing = torch.div(index_diff[0], gcd, rounding_mode="floor").item()
        t_indices = torch.arange(0, common_length, t_spacing, dtype=torch.long)
        x_indices = torch.arange(0, common_length, x_spacing, dtype=torch.long)
        tx_indices = torch.LongTensor(np.union1d(t_indices, x_indices))

        # compute integrand with interpolated values
        r_int = torch.from_numpy(
            interp1d(x_indices, r_eval, axis=0, kind="linear", assume_sorted=True)(
                tx_indices
            )
        )
        q_int = torch.from_numpy(
            interp1d(t_indices, q_eval, axis=0, kind="linear", assume_sorted=True)(
                tx_indices
            )
        )
        ksi_diff = index_diff[1] / index_diff[0]
        integrand = torch.sum((q_int - torch.sqrt(ksi_diff) * r_int) ** 2, dim=-1)

    start_t = get_state(state_index=state_index, data=data)[0]
    end_t = get_state(state_index=next_state_index, data=data)[0]

    # compute integral
    r_int = trapezoid(integrand, tx_indices) * (end_t - start_t) / tx_indices[-1]
    return r_int


def r_cost_old(
    state_index: torch.LongTensor, next_state_index: torch.LongTensor, data: dataset
) -> torch.Tensor:
    assert torch.all(next_state_index >= state_index), "wrong direction"
    if torch.all(state_index == next_state_index):
        return zero_
    # elif torch.any(state_index == next_state_index):
    #     return penalty_

    t_data, q_data, r_data = data[:]

    index_diff = next_state_index - state_index
    gcd = torch.gcd(index_diff[..., 0], index_diff[..., 1])

    # one extra since we want the end state included
    index_length = (
        torch.div(torch.prod(index_diff).long(), gcd, rounding_mode="floor") + 1
    )

    # make indexes that repeats itself x_scale times
    # t
    indexes_t = torch.linspace(state_index[0], next_state_index[0], index_length).long()
    # x
    indexes_x = torch.linspace(state_index[1], next_state_index[1], index_length).long()

    # corresponding t values
    start_t = get_state(state_index=state_index, data=data)[0].detach()
    end_t = get_state(state_index=next_state_index, data=data)[0].detach()

    t_eval = torch.linspace(
        start_t,
        end_t,
        steps=index_length,
    )
    q_eval = q_data[indexes_t]
    r_eval = r_data[indexes_x]

    ksi_diff = index_diff[1] / index_diff[0]
    integrand = torch.sum((q_eval - torch.sqrt(ksi_diff) * r_eval) ** 2, dim=-1)
    r_int = torch.trapezoid(integrand, t_eval)

    return r_int


def get_epsilon_greedy(epsilon: float, num_actions: int) -> callable:
    def epsilon_greedy(
        state_index: torch.LongTensor, model: callable, data: dataset
    ) -> torch.LongTensor:
        if torch.rand(1) < epsilon:
            return torch.randint(low=0, high=num_actions, size=(1,)).long()
        else:
            return get_optimal_action(q_model=model, state_index=state_index, data=data)

    return epsilon_greedy


def get_state(state_index: torch.LongTensor, data: dataset):
    t_data = data[:][0]
    return t_data[state_index]


def is_end_state(state_index: torch.LongTensor, data: dataset):
    t_data = data[:][0]
    grid_max = t_data.size(0)
    if torch.all(state_index == grid_max - 1):
        return True


def get_optimal_action(
    q_model: callable, state_index: torch.LongTensor, data: dataset
) -> torch.LongTensor:
    state = get_state(state_index=state_index, data=data)
    action_index = torch.argmin(q_model(state), dim=-1).unsqueeze(-1).long()
    return action_index


def get_action_map(depth, data):
    # assumes same base
    # hack to make a list of all admissible directions (in reverse)
    grid_size = data[:][0].size(0)
    assert depth <= grid_size, "action space larger than state space"
    action_array = torch.LongTensor(
        np.indices((depth, depth)).T.reshape(depth ** 2, 2) + 1
    )

    # could also have used np.unravel_index
    def action_map(
        action_index: torch.LongTensor, state_index: torch.LongTensor
    ) -> torch.LongTensor:
        """state is index (i, j), action is (i)"""

        # torch.LongTensor(np.divmod(action_index, base_dim)[-1])

        # It is strange that actions act on state indices, but it should work
        action = action_array[action_index[..., 0]]
        new_state_index = torch.LongTensor(
            np.minimum(state_index + action, grid_size - 1)
        )

        return new_state_index

    return action_map


def get_optimal_path(
    q_model: callable,
    action_map,
    data=dataset,
    start_state_index=torch.LongTensor([[0, 0]]),
) -> torch.LongTensor:
    # q_model: callable, state_index: torch.LongTensor, data: dataset
    # no beautiful recursion here, python is stupid :(
    state_index = start_state_index.detach().clone()
    index_list = [state_index]
    while not is_end_state(state_index, data):
        action_index = get_optimal_action(
            q_model=q_model, state_index=state_index, data=data
        )

        # update
        state_index = action_map(action_index=action_index, state_index=state_index)

        # store next
        index_list.append(state_index)
    return torch.vstack(index_list)


def get_path_value(path: torch.LongTensor, data: dataset):
    value = torch.zeros(1)
    # print(path)
    for i in range(len(path) - 1):
        value += r_cost(state_index=path[i], next_state_index=path[i + 1], data=data)
    return value


def plot_solution(q_model, ksi, data, action_map):
    with torch.no_grad():
        x_eval, *_ = data[:]

        N = len(data)
        grid = x_eval[np.indices((N, N)).T]

        cost_matrix = torch.min(q_model(grid).detach(), dim=-1)[0]
        # print(grid, cost_matrix)
        indexes = get_optimal_path(q_model=q_model, action_map=action_map, data=data)

        fig, ax = plt.subplots(1)
        plot = ax.imshow(cost_matrix, extent=[0, 1, 0, 1], origin="lower")
        plt.colorbar(plot)

        ax.scatter(x_eval[indexes[:, 0]], x_eval[indexes[:, 1]], label="DQN")
        ksi_eval = ksi(x_eval.unsqueeze(1))
        ax.plot(x_eval, ksi_eval, label="true")
        plt.legend()

        plt.show()

    return ax


def compute_loss_rl(model: callable, data: dataset, action_map):
    path = get_optimal_path(q_model=model, action_map=action_map, data=data)
    return get_path_value(path=path, data=data)


class RLSystem:
    def __init__(
        self,
        action_map: callable,
        get_optimal_action: callable,
        r_cost: callable,
        is_end_state,
        choose_action,
        get_state: callable,
        start_state,
    ):
        pass


def fit_dqn_deterministic(
    model: callable,
    data: dataset,
    num_epochs,
    optimizer,
    start_state_index,
    get_state: callable,
    choose_action: callable,
    action_map: callable,
    is_end_state: callable,
    init: callable = None,
    data_val=None,
    track_history=True,
    verbose=False,
    verbose_interval=100,
    learning_rate=None,
    init_weight_seed: int = None,
    compute_loss: callable = compute_loss_rl,
    max_nan_steps=50,
    **kwargs
) -> tuple[callable, torch.Tensor, torch.Tensor]:
    if init is not None:
        init(model, init_weight_seed=init_weight_seed)

    if learning_rate is None:
        learning_rate = larning_rates[optimizer]
    # select optimizer
    if optimizer == "ADAM":
        optimizer_ = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "SGD":
        optimizer_ = optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "LBFGS":
        optimizer_ = optim.LBFGS(
            model.parameters(),
            max_iter=1,
            max_eval=50000,
            tolerance_change=1.0 * np.finfo(float).eps,
            lr=learning_rate,
        )
    elif optimizer == "strong_wolfe":
        optimizer_ = optim.LBFGS(
            model.parameters(),
            lr=learning_rate,
            max_iter=100,
            max_eval=1000,
            history_size=200,
            line_search_fn="strong_wolfe",
        )
        max_nan_steps = 2
        verbose_interval = 1 if num_epochs < 10 else 5
    elif callable(optimizer):
        optimizer_ = optimizer(model.parameters)
    else:
        raise ValueError("Optimizer not recognized")

    loss_history_train = torch.zeros(num_epochs)
    nan_steps = 0
    # Loop over epochs
    for epoch in range(num_epochs):
        # make indexes that repeats itself t_scale times
        if verbose and not epoch % verbose_interval:
            print(
                "################################ ",
                epoch,
                " ################################",
            )

        state_index = start_state_index
        # start_state = get_state(state_index=state_index, data=data)
        # print("pred_loss: ", torch.min(model(start_state)).item())
        # path = get_optimal_path(
        #     model, action_map=action_map, data=data,
        #     start_state_index=start_state_index
        # )
        # print("policy_loss: ", get_path_value(path=path, data=data).item())
        while not is_end_state(state_index, data):
            action_index = choose_action(
                state_index=state_index, model=model, data=data
            )
            # print(action_index)
            next_state_index = action_map(
                action_index=action_index, state_index=state_index
            )

            # print("states: ", state_index, next_state_index)

            def closure():
                # zero the parameter gradients
                optimizer_.zero_grad()
                # forward + backward + optimize

                state = get_state(state_index=state_index, data=data)
                next_state = get_state(state_index=next_state_index, data=data)

                Q_optim = torch.min(model(state), dim=-1)[0]

                with torch.no_grad():
                    # make sure start state has 0  value
                    Q_next_est = (
                        torch.min(model(next_state), dim=-1)[0]
                        if not is_end_state(next_state_index, data)
                        else zero_
                    )
                    r_eval = r_cost(
                        state_index=state_index,
                        next_state_index=next_state_index,
                        data=data,
                    )
                    assert r_eval >= 0, "negative cost"
                    Q_est = Q_next_est + r_eval

                loss = l2_loss(Q_optim, Q_est)
                loss.backward()
                return loss

            optimizer_.step(closure=closure)

            state_index = next_state_index
        if track_history:
            train_loss = compute_loss(
                model=model, action_map=action_map, data=data
            ).detach()

            loss_history_train[epoch] = train_loss
            if track_history:
                # stop if nan output
                if torch.isnan(train_loss):
                    nan_steps += 1
                if epoch % 100 == 0:
                    nan_steps = 0

        if verbose and not epoch % verbose_interval and track_history:
            print("Training Loss: ", np.round(loss_history_train[epoch], 8))

        if nan_steps > max_nan_steps:
            break

    if verbose and track_history and len(loss_history_train) > 0:
        print("Final training Loss: ", np.round(loss_history_train[-1], 8))

    return (
        model,
        torch.as_tensor(loss_history_train),
        torch.as_tensor(loss_history_train),
    )
