import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from deepthermal.FFNN_model import larning_rates

from neural_reparam.reparam_env import DiscreteReparamEnv, compute_loss_rl

State = torch.LongTensor

# penalty = 1er
# penalty_ = torch.tensor(penalty)
zero_ = torch.tensor(0.0)
l2_loss = nn.MSELoss()


def fit_dqn_deterministic(
    model: callable,
    num_epochs,
    optimizer,
    batch_size: int,
    choose_action: callable,
    env: DiscreteReparamEnv,
    init: callable = None,
    track_history=True,
    verbose=False,
    verbose_interval=100,
    learning_rate=None,
    gamma: float = 1.0,
    init_weight_seed: int = None,
    compute_loss: callable = compute_loss_rl,
    max_nan_steps=50,
    C: int = 10,
    memory_size: int = 20,
    DDQN: bool = False,
    lr_scheduler=None,
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

    if lr_scheduler is not None:
        scheduler = lr_scheduler(optimizer_)

    loss_history_train = torch.zeros(num_epochs)
    nan_steps = 0

    # initialize momory
    mem_action_indexes = env.sample_action(env=env, N=memory_size)
    mem_state_indexes = env.sample_states(env=env, N=memory_size)
    mem_r_costs = torch.zeros((memory_size, 1))
    for i in range(memory_size):
        next_state_index = env.action_map(
            action_index=mem_action_indexes[i], state_index=mem_state_indexes[i]
        )
        mem_r_costs[i] = env.r_cost(
            state_index=mem_state_indexes[i],
            next_state_index=next_state_index,
            env=env,
        )

    # memory is reference to data that will be updated
    replay_memory = TensorDataset(mem_state_indexes, mem_action_indexes, mem_r_costs)
    memory_iter = 0

    # Loop over epochs
    for epoch in range(num_epochs):
        # make indexes that repeats itself t_scale times
        if verbose and not epoch % verbose_interval:
            print(
                "################################ ",
                epoch,
                " ################################",
            )
        # init epoch
        state_index = env.start_state_index
        C_iter = 0
        model_hat = copy.deepcopy(model)
        while not env.is_end_state(state_index, env=env):

            if C_iter == C:
                model_hat = copy.deepcopy(model)

                C_iter = 0
            C_iter += 1
            memory_iter = (memory_iter + 1) % memory_size

            #  get next state and update memory
            with torch.no_grad():
                action_index = choose_action(
                    state_index=state_index, model=model, env=env
                )

                next_state_index = env.action_map(
                    action_index=action_index, state_index=state_index
                )
                mem_action_indexes[memory_iter] = action_index
                mem_state_indexes[memory_iter] = state_index
                mem_r_costs[memory_iter] = env.r_cost(
                    state_index=state_index, next_state_index=next_state_index, env=env
                )

            # get data from memory
            train_data = DataLoader(
                replay_memory, batch_size=batch_size, shuffle=True, drop_last=False
            )

            for state_indexes_i, action_indexes_i, r_costs_i in train_data:
                next_state_indexes_i = env.action_map(
                    action_index=action_indexes_i, state_index=state_indexes_i
                )

                states = env.get_state(state_index=state_indexes_i, env=env)
                next_states = env.get_state(state_index=next_state_indexes_i, env=env)

                def closure():
                    # zero the parameter gradients
                    optimizer_.zero_grad()
                    # forward + backward + optimize

                    with torch.no_grad():
                        # make sure start state has 0  value
                        # print(next_state_indexes_i, )
                        if DDQN:
                            actions = torch.argmin(
                                model(next_states), dim=-1, keepdim=True
                            )
                            Q_hat_next_no_bound = torch.gather(
                                model_hat(next_states), dim=-1, index=actions
                            )

                        else:
                            Q_hat_next_no_bound = torch.min(
                                model_hat(next_states), dim=-1, keepdim=True
                            )[0]

                        Q_hat_next = torch.where(
                            env.is_end_state(next_state_indexes_i, env=env),
                            torch.tensor(0, dtype=torch.float),
                            Q_hat_next_no_bound,
                        )
                        Y = gamma * Q_hat_next + r_costs_i
                        # Y = torch.where(env.is_start_state(state_indexes_i, env= env),
                        #                 torch.tensor(0, dtype=torch.float),
                        #                 gamma * Q_hat_next_no_bound + r_costs_i)
                    Q_optim = torch.gather(
                        model(states), dim=-1, index=action_indexes_i
                    )

                    loss = l2_loss(Q_optim, Y)
                    loss.backward()
                    return loss

                optimizer_.step(closure=closure)
                break

            state_index = next_state_index
        if track_history or (lr_scheduler is not None):
            train_loss = compute_loss(model=model, env=env).detach()

            if lr_scheduler is not None:
                scheduler.step(train_loss)

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
