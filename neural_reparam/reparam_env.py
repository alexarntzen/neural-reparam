import numpy as np
from itertools import product
from math import gcd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import dataset
from typing import Union

from deepthermal.plotting import plot_result_sorted

State = torch.LongTensor

zero_ = torch.tensor(0.0)
l2_loss = nn.MSELoss()


class Env:
    def __init__(self, data: dataset, depth: int = 4):
        self.data = data
        self.N = data[:][0].size(0)
        self.get_state = get_state
        self.sample_states = sample_states
        self.sample_action = sample_action

        self.action_map = None
        self.num_actions = None
        self.is_start_state = None
        self.is_end_state = None
        self.r_cost = None
        self.start_state_index = None


class DiscreteReparamEnv(Env):
    def __init__(self, data: dataset, depth: int = 4):
        super().__init__(data=data, depth=depth)
        self.action_map, self.num_actions = get_action_map(depth, N=self.N)
        self.is_start_state = is_start_state
        self.is_end_state = is_end_state
        self.get_state = get_state
        self.r_cost = r_cost
        self.start_state_index = torch.LongTensor([0, 0])


class DiscreteReparamReverseEnv(Env):
    def __init__(self, data: dataset, depth: int):
        super().__init__(data=data, depth=depth)
        self.action_map, self.num_actions = get_action_map(
            depth, N=self.N, reverse=True
        )
        self.is_start_state = is_end_state
        self.is_end_state = is_start_state
        self.r_cost = r_cost_reverse
        self.start_state_index = torch.LongTensor([self.N - 1, self.N - 1])


@torch.no_grad()
def r_cost(
    state_index: torch.LongTensor, next_state_index: torch.LongTensor, env: Env
) -> torch.Tensor:
    assert torch.all(next_state_index >= state_index), "wrong direction"
    if state_index[0] == next_state_index[0]:
        return zero_
    # elif torch.any(state_index == next_state_index):
    #     return penalty_

    t_data, q_data, r_data = env.data[:]
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

    start_t = env.get_state(state_index=state_index, env=env)[0]
    end_t = env.get_state(state_index=next_state_index, env=env)[0]

    # compute integral
    r_int = trapezoid(integrand, tx_indices) * (end_t - start_t) / tx_indices[-1]
    return r_int


def r_cost_reverse(
    state_index: torch.LongTensor, next_state_index: torch.LongTensor, env: Env
) -> torch.Tensor:
    return r_cost(state_index=next_state_index, next_state_index=state_index, env=env)


def get_epsilon_greedy(epsilon: float, num_actions: int) -> callable:
    def epsilon_greedy(
        state_index: torch.LongTensor, model: callable, env: Env
    ) -> torch.LongTensor:
        if torch.rand(1) < epsilon:
            return torch.randint(low=0, high=num_actions, size=(1,)).long()
        else:
            return get_optimal_action(q_model=model, state_index=state_index, env=env)

    return epsilon_greedy


def get_state(state_index: torch.LongTensor, env: Env):
    """works also with multiple states"""
    t_data = env.data[:][0]
    return t_data[state_index].float()


def is_end_state(state_index: torch.LongTensor, env: Env):
    t_data = env.data[:][0]
    grid_max_index = t_data.size(0) - 1
    answer = torch.logical_and(
        state_index[..., 0:1] == grid_max_index, state_index[..., 1:2] == grid_max_index
    )
    return answer


def is_start_state(state_index: torch.LongTensor, env: dataset):
    return torch.logical_and(state_index[..., 0:1] == 0, state_index[..., 1:2] == 0)


def get_optimal_action(
    q_model: callable, state_index: torch.LongTensor, env: dataset
) -> torch.LongTensor:
    state = get_state(state_index=state_index, env=env).float()
    action_index = torch.argmin(q_model(state), dim=-1, keepdim=True).long()
    return action_index


def get_action_map(depth, N: int, reverse: bool = False):
    # assumes same base
    # hack to make a list of all admissible directions (in reverse)
    max_grid_index = torch.tensor(N - 1).long()
    assert depth <= max_grid_index, "action space larger than state space"
    action_array = torch.LongTensor(
        [
            [x, y]
            for x, y in product(range(1, depth + 1), range(1, depth + 1))
            if gcd(x, y) == 1
        ]
    )
    if reverse:
        action_array *= -1

    # could also have used np.unravel_index
    def action_map(
        action_index: torch.LongTensor, state_index: torch.LongTensor
    ) -> torch.LongTensor:
        """state is index (i, j), action is (i)

        Works with many actions"""

        # torch.LongTensor(np.divmod(action_index, base_dim)[-1])

        # It is strange that actions act on state indices, but it should work
        action = action_array[action_index[..., 0]]
        new_state_index = torch.clamp(
            state_index + action, min=0, max=max_grid_index
        ).long()
        return new_state_index

    return action_map, len(action_array)


def get_optimal_path(
    q_model: callable,
    env: Env,
) -> Union[torch.LongTensor, None]:
    # q_model: callable, state_index: torch.LongTensor, data: dataset
    # no beautiful recursion here, python is stupid :(
    state_index = env.start_state_index.detach().clone()
    t_data = env.data[:][0]
    N, dim = t_data.size(0), len(state_index.flatten())
    index_tensor = torch.zeros((2 * N, dim), dtype=torch.long)
    index_tensor[0] = state_index
    for i in range(1, 2 * N):

        # return if found total
        if env.is_end_state(state_index, env=env):
            return index_tensor[:i]

        action_index = get_optimal_action(
            q_model=q_model, state_index=state_index, env=env
        )

        # update
        state_index = env.action_map(action_index=action_index, state_index=state_index)

        # store next
        index_tensor[i] = state_index
    return None


def get_path_value(path: torch.LongTensor, env: Env) -> torch.Tensor:
    if path is None:
        return torch.tensor(torch.inf)
    value = torch.zeros(1)
    for i in range(len(path) - 1):
        value += env.r_cost(state_index=path[i], next_state_index=path[i + 1], env=env)
    return value


def compute_loss_rl(model: callable, env: Env):
    path = get_optimal_path(q_model=model, env=env)
    return get_path_value(path=path, env=env)


def sample_states(env: Env, N: int = 1) -> State:
    grid_len = len(env.data)
    states = torch.randint(0, grid_len, size=(N, 1), dtype=torch.long).long()
    return torch.cat((states, states), dim=1)


def sample_action(
    env: Env,
    N: int = 1,
) -> torch.LongTensor:
    actions = torch.randint(0, env.num_actions, size=(N, 1), dtype=torch.long).long()
    return actions


@torch.no_grad()
def plot_solution_rl(model, env: Env, **kwargs):
    x_eval, *_ = env.data[:]
    N = len(x_eval)
    ind = torch.as_tensor(np.indices((N, N)).T)

    grid = x_eval[ind]

    # comptue cost
    cost_matrix = torch.min(model(grid).detach(), dim=-1)[0]
    computed_path_indexes = get_optimal_path(q_model=model, env=env)
    computed_path = x_eval[computed_path_indexes]

    # add V values to axes
    fig, ax = plt.subplots(1)
    plot = ax.imshow(cost_matrix, extent=[0, 1, 0, 1], origin="lower")
    fig.colorbar(plot)

    plot_result_sorted(
        x_pred=computed_path[:, 0], y_pred=computed_path[:, 1], fig=fig, **kwargs
    )
    return fig
