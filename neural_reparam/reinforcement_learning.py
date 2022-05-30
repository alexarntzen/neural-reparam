import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from tqdm import tqdm

from deepthermal.FFNN_model import larning_rates

from neural_reparam.reparam_env import DiscreteReparamEnv, ReparamEnv

State = torch.LongTensor

# penalty = 1er
# penalty_ = torch.tensor(penalty)
zero_ = torch.tensor(0.0)
l2_loss = nn.MSELoss()
smooth_l1 = nn.SmoothL1Loss()


def fit_dqn_deterministic(
    get_env: callable,
    model: callable,
    num_epochs,
    optimizer,
    batch_size: int,
    init: callable = None,
    track_history=True,
    verbose=False,
    verbose_interval=100,
    learning_rate=None,
    gamma: float = 1.0,
    init_weight_seed: int = None,
    max_nan_steps=50,
    update_every: int = 10,
    max_ep_len: int = None,
    initial_steps: int = 1000,
    memory_size: int = 20,
    epsilon: float = 0.01,
    DDQN: bool = False,
    lr_scheduler=None,
    **kwargs,
) -> tuple[callable, np.ndarray, np.ndarray]:
    """"""
    if init is not None:
        init(model, init_weight_seed=init_weight_seed)

    # model = critic, is updated
    # model_actor =  actor
    model_actor = copy.deepcopy(model)

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
        optimizer_ = optimizer(model.parameters())
    else:
        raise ValueError("Optimizer not recognized")

    if lr_scheduler is not None:
        scheduler = lr_scheduler(optimizer_)

    q_loss_history = torch.zeros(num_epochs)
    rewards_history = torch.zeros(num_epochs)

    nan_steps = 0

    # initialize env
    env: DiscreteReparamEnv = get_env()
    env.reset()
    if max_ep_len is None:
        max_ep_len = env.size * 2

    # init replay memory
    replay_memory = ReplayMemory(memory_size=memory_size, env=env)

    # init memory
    for i in range(initial_steps):
        # a = env.action_space.sample()
        a = get_optimal_action(model_actor, env=env)
        o1 = env.state
        o2, r, d, _ = env.step(action=a)

        po1, po2 = preprocess_states(o1, o2, env=env)
        replay_memory.push(po1, a, po2, r, d)
        if d:
            env.reset()

    # Loop over epochs
    total_steps = 0
    for epoch in tqdm(
        range(num_epochs), desc="Epoch: ", disable=(not verbose), leave=False
    ):
        try:
            # init epoch/episode
            env.reset()
            done = False
            q_losses = torch.zeros(max_ep_len)
            steps = 0

            # take steps
            for step in range(max_ep_len):
                action = epsilon_greedy(model_actor, env=env, epsilon=epsilon)
                state = env.state
                next_state, reward, done, _ = env.step(action=a)
                steps += 1
                total_steps += 1
                po1, po2 = preprocess_states(state, next_state, env=env)
                replay_memory.push(po1, action, po2, reward, done)

                # get data from memory
                (
                    po1_batch,
                    a_batch,
                    po2_batch,
                    reward_batch,
                    done_batch,
                ) = replay_memory.sample(batch_size)

                def closure():
                    # zero the parameter gradients
                    optimizer_.zero_grad()
                    # forward + backward + optimize

                    with torch.no_grad():
                        # make sure start state has 0  value
                        # print(next_state_indexes_i, )
                        if DDQN:
                            actions = torch.argmax(
                                model(po2_batch), dim=-1, keepdim=True
                            )
                            Q_hat_next = torch.gather(
                                model_actor(po2_batch), dim=-1, index=actions
                            )

                        else:
                            Q_hat_next = torch.max(
                                model_actor(po2_batch), dim=-1, keepdim=True
                            )[0]

                        Y = gamma * Q_hat_next * done_batch + reward_batch

                    Q_optim = torch.gather(model(po1_batch), dim=-1, index=a_batch)
                    # Compute Huber loss

                    loss = smooth_l1(Q_optim, Y)
                    loss.backward()
                    q_losses[steps] = loss.item()
                    return loss
                    # end closure

                # update actor
                if steps % update_every == 0:
                    model_actor.load_state_dict(model.state_dict())

                optimizer_.step(closure=closure)
                if done:
                    break
                # end step

            if track_history or (lr_scheduler is not None):
                epoch_q_loss = torch.sum(q_losses) / steps
                rewards_history[epoch] = get_value(model=model, env=env)
                q_loss_history[epoch] = epoch_q_loss

                if lr_scheduler is not None:
                    scheduler.step(epoch_q_loss)

                if track_history:
                    # stop if nan output
                    if torch.isnan(epoch_q_loss):
                        nan_steps += 1
                    if epoch % 100 == 0:
                        nan_steps = 0

            if verbose and not epoch % verbose_interval and track_history:
                print("Reward: ", np.round(rewards_history[epoch], 8))
                print("Q loss: ", np.round(q_loss_history[epoch], 8))

            if nan_steps > max_nan_steps:
                break
            # end epoch

        except KeyboardInterrupt:
            print("Interrupted breaking")
            break
        # end epochs

    if verbose and track_history:
        if len(rewards_history) > 0:
            print("Final eward: ", np.round(rewards_history[-1], 8))
        if len(q_loss_history) > 0:
            print("Final Q loss: ", np.round(q_loss_history[-1], 8))

    return model, np.array(q_loss_history), np.array(rewards_history)


class ReplayMemory(object):
    def __init__(self, memory_size, env: ReparamEnv):
        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape
        self.o1_array = torch.zeros([memory_size, *obs_dim], dtype=torch.float32)
        self.a_array = torch.zeros([memory_size, *act_dim], dtype=torch.int64)
        self.o2_array = torch.zeros([memory_size, *obs_dim], dtype=torch.float32)
        self.r_array = torch.zeros([memory_size, 1], dtype=torch.float32)
        self.d_array = torch.zeros([memory_size, 1], dtype=torch.int)
        self.ptr, self.size, self.max_size = 0, 0, memory_size

    def push(self, o1, a, o2, r, d):
        self.o1_array[self.ptr] = o1
        self.a_array[self.ptr] = a
        self.o2_array[self.ptr] = o2
        self.r_array[self.ptr] = r
        self.d_array[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.o1_array[idxs],
            self.a_array[idxs],
            self.o2_array[idxs],
            self.r_array[idxs],
            self.d_array[idxs],
        )

    def __len__(self):
        return self.size


@torch.no_grad()
def get_optimal_action(
    model: callable, env: ReparamEnv, state: gym.core.ObsType = None
) -> torch.LongTensor:
    if state is None:
        state = env.state
    p_state = torch.as_tensor(env.preprocess_state(state), dtype=torch.float32)
    action_index = torch.argmin(model(p_state), dim=-1, keepdim=True).long()
    return action_index


def get_value(
    model: callable,
    env: ReparamEnv,
    start_state: gym.core.ObsType = None,
    max_ep_len: int = None,
) -> float:
    o1, d, value = env.reset(), False, 0
    if max_ep_len is None:
        max_ep_len = env.size**2

    if start_state is None:
        env.state = o1
    else:
        env.state = start_state
        o1 = start_state
    a = get_optimal_action(model=model, state=o1, env=env)

    for i in range(max_ep_len):
        o2, r, d, _ = env.step(action=a)
        value += r
        o1 = o2

        if d:
            break
    return value


def get_path_value(path: torch.LongTensor, env: ReparamEnv) -> torch.Tensor:
    if path is None:
        return torch.tensor(torch.inf)
    value = 0

    for i in range(len(path) - 1):
        value += env._r_cost(path[i], path[i + 1])
    return value


def get_epsilon_greedy(epsilon: float, num_actions: int) -> callable:
    def epsilon_greedy(model: callable, env: ReparamEnv) -> int:
        if torch.rand(1) < epsilon:
            return env.action_space.sample()
        else:
            return get_optimal_action(model=model, env=env)

    return epsilon_greedy


def epsilon_greedy(model: callable, env: ReparamEnv, epsilon: float = 0.01) -> int:
    if torch.rand(1) < epsilon:
        return env.action_space.sample().item()
    else:
        return get_optimal_action(model=model, env=env)


def preprocess_states(*states, env: ReparamEnv):
    for state in states:
        yield torch.as_tensor(env.preprocess_state(state), dtype=torch.float32)
