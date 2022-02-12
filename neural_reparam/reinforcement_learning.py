import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import dataset

from deepthermal.FFNN_model import compute_loss_torch, larning_rates

penalty = 1e0
penalty_ = torch.LongTensor([penalty])
zero_ = torch.zeros(1)
l2_loss = nn.MSELoss()


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
    loss_func=nn.MSELoss(),
    compute_loss: callable = compute_loss_torch,
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

    loss_history_train = list()
    loss_history_val = list()
    nan_steps = 0
    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose and not epoch % verbose_interval:
            print(
                "################################ ",
                epoch,
                " ################################",
            )

        state_index = start_state_index

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
                    Q_est = Q_next_est + r_eval

                # would not need abs here but sometimes it goes negative
                # print("qopt, qest", Q_optim, Q_next_est, r_eval)
                loss = l2_loss(Q_optim, Q_est)
                loss.backward()
                return loss

            optimizer_.step(closure=closure)
            state_index = next_state_index
        track_history = False
        if track_history:
            x_train, y_train = data[:]
            train_loss = compute_loss(
                loss_func=loss_func,
                model=model,
                x_train=x_train,
                y_train=y_train,
            ).detach()
            if track_history:
                # stop if nan output
                if torch.isnan(train_loss):
                    nan_steps += 1
                if epoch % 100 == 0:
                    nan_steps = 0

        if verbose and not epoch % verbose_interval and track_history:
            print_iter = -1
            print("Training Loss: ", np.round(loss_history_train[print_iter], 8))
            if data_val is not None and len(data_val) > 0:
                print("Validation Loss: ", np.round(loss_history_val[print_iter], 8))

        if nan_steps > max_nan_steps:
            break

    if verbose and track_history and len(loss_history_train) > 0:
        print("Final training Loss: ", np.round(loss_history_train[-1], 8))
        if data_val is not None and len(data_val) > 0:
            print("Final validation Loss: ", np.round(loss_history_val[-1], 8))

    return model, torch.as_tensor(loss_history_train), torch.as_tensor(loss_history_val)


def r_cost(
    state_index: torch.LongTensor, next_state_index: torch.LongTensor, data: dataset
) -> torch.Tensor:
    # TODO: double check this
    # assumes eqally spaced points
    assert torch.all(next_state_index >= state_index), "wrong direction"
    if torch.all(state_index == next_state_index):
        return zero_
    elif torch.any(state_index == next_state_index):
        return penalty_

    t_data, q_data, r_data = data[:]
    state_diff = get_state(state_index=next_state_index, data=data) - get_state(
        state_index=state_index, data=data
    )
    grid_size = t_data.size(0)
    # one extra since we want the end state included
    index_diff = 1 + next_state_index - state_index
    gcd = torch.gcd(index_diff[0], index_diff[1])
    len_longest_subseq = torch.div(
        index_diff[0] * index_diff[1], gcd ** 2, rounding_mode="floor"
    )

    # make on same length
    # t
    if state_diff[0] < len_longest_subseq:
        diff_indexes_t = torch.div(
            torch.arange(len_longest_subseq) * index_diff[0],
            len_longest_subseq,
            rounding_mode="floor",
        )
    else:
        diff_indexes_t = torch.arange(index_diff[0])
    indexes_t = diff_indexes_t + state_index[0]

    # x
    if state_diff[1] < len_longest_subseq:
        diff_indexes_x = torch.div(
            torch.arange(len_longest_subseq) * index_diff[1],
            len_longest_subseq,
            rounding_mode="floor",
        )
    else:
        diff_indexes_x = torch.arange(index_diff[1])
    indexes_x = diff_indexes_x + state_index[1]

    # make sure to only take integral inside the
    assert torch.max(indexes_x) < grid_size, "index out of bounds"
    assert torch.max(indexes_t) < grid_size, "index out of bounds"

    q_eval = q_data[indexes_t]
    r_eval = r_data[indexes_x]
    ksi_diff = index_diff[1] / index_diff[0]

    left = q_eval
    right = torch.sqrt(ksi_diff) * r_eval
    r_int = state_diff[0] ** 2 * l2_loss(left, right)
    return r_int


def get_epsilon_greedy(epsilon: float, num_actions: int) -> callable:
    def epsilon_greedy(
        state_index: torch.LongTensor, model: callable, data: dataset
    ) -> torch.LongTensor:
        if torch.rand(1) < epsilon:
            return torch.LongTensor(torch.randint(low=0, high=num_actions, size=(1,)))
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
    action_index = torch.LongTensor(torch.argmin(q_model(state), dim=-1).unsqueeze(-1))
    return action_index


def get_action_map(depth, data):
    # assumes same base
    # hack to make a list of all admissible directions (in reverse)
    grid_size = data[:][0].size(0)
    assert depth ** 2 <= grid_size, "action space larger than state space"
    action_array = torch.IntTensor(
        np.indices((depth, depth)).T.reshape(depth ** 2, 2) + 1
    )

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
