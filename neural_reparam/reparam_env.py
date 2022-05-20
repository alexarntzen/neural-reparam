import numpy as np
from itertools import product
from math import gcd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from typing import Union

from deepthermal.plotting import plot_result_sorted

import gym
from gym import spaces
from typing import Optional

State = torch.LongTensor

zero_ = torch.tensor(0.0)
l2_loss = nn.MSELoss()


class ReparamEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        q_func: callable = None,
        r_func: callable = None,
        data: Dataset = None,
        size: int = None,
    ):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert (
            q_func is not None and r_func is not None and size is not None
        ) or data is not None

        if data is not None:
            self.data = data
            self.t_data, self.q_data, self.r_data = data[:]
            self.size = len(self.t_data)

        if q_func is not None and r_func is not None:
            self.q = q_func
            self.r = r_func
            self.size = size
            if data is None:
                self.t_data = torch.linspace(0, 1, size)
                self.q_data = q_func(self.t_data)
                self.r_data = r_func(self.t_data)
                self.data = torch.utils.data.TensorDataset(
                    self.t_data, self.q_data, self.r_data
                )
        else:
            self.q = interp1d(
                y=self.q_data,
                x=self.t_data.flatten().detach().numpy(),
                axis=0,
                assume_sorted=True,
            )
            self.r = interp1d(
                y=self.r_data,
                x=self.t_data.flatten().detach().numpy(),
                axis=0,
                assume_sorted=True,
            )

        self.rgb_array_white = np.empty([self.size, self.size, 3], dtype=int)
        self.rgb_array_white.fill(255)

    def reset(
        self, seed=None, return_info=False, options=None
    ) -> Union[gym.core.ObsType, tuple[gym.core.ObsType, dict]]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.rgb_array = np.copy(self.rgb_array_white)

        self.state = self.start_state
        observation = self.state

        if isinstance(observation, torch.Tensor):
            observation = observation.numpy()
        return (observation, dict()) if return_info else observation

    def step(
        self, action: gym.core.ActType
    ) -> tuple[gym.core.ObsType, float, bool, dict]:
        # update state
        new_state = self._action_map(action=action, state=self.state)

        # An episode is done if the agent has reached the target
        done = self._is_end_state(state=new_state)
        reward = self._r_cost(state=self.state, next_state=new_state)
        observation = new_state

        self.state = new_state
        if isinstance(observation, torch.Tensor):
            observation = observation.numpy()
        return observation, reward, done, dict()

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            self.rgb_array[self.get_state_index] = 0
            return self.rgb_array
        else:
            super(ReparamEnv, self).render(mode=mode)  # just raise an exception


class RealReparamEnv(ReparamEnv):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        q_func: callable = None,
        r_func: callable = None,
        data: Dataset = None,
        size: int = None,
    ):
        super().__init__(
            render_mode=render_mode, q_func=q_func, r_func=r_func, data=data, size=size
        )

        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.reward_range = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=np.float32)

        self.end_state = np.ones(2, dtype=np.float32)
        self.start_state = np.zeros(2, dtype=np.float32)

        self.reset()

    def _action_map(self, action, state):
        return np.clip(self.state + action, 0, 1)

    def _is_end_state(self, state):
        return np.array_equal(state, self.end_state)

    def get_state_index(self) -> tuple:
        return round(self.state[0] * self.size), round(self.state[1] * self.size)

    def _r_cost(self, state: np.ndarray, next_state: np.ndarray) -> np.float32:
        """Probably could be more efficient"""
        # extract values
        t_0, x_0 = state
        t_1, x_1 = next_state
        dt, dx = next_state - state
        if dt == 0:
            return 0

        num_integration_points = round(2 * max(dt, dx) * self.size)

        # make a list of values to integrate
        t_values = np.linspace(t_0, t_1, num_integration_points, dtype=np.float32)
        x_values = np.linspace(x_0, x_1, num_integration_points, dtype=np.float32)

        # calculate the shifted r and q values
        r_shifted_values = self.r(x_values)
        q_values = self.q(t_values)

        # calculate values in the integrand
        integrand = np.sum(
            (q_values - r_shifted_values * np.sqrt(dx / dt)) ** 2, axis=-1
        )
        return trapezoid(integrand, t_values).item()


class DiscreteReparamEnv(ReparamEnv):
    def __init__(self, depth: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.get_state = get_state
        self.sample_states = sample_states
        self.sample_action = sample_action

        self.action_map, self.num_actions = get_action_map(depth, size=self.size)
        self.is_start_state = is_start_state
        self.is_end_state = is_end_state
        self.get_state = get_state
        self.r_cost = r_cost
        self.start_state = torch.LongTensor([0, 0])

        self.observation_space = spaces.Box(
            low=0, high=self.size, shape=(2,), dtype=int
        )
        self.action_space = spaces.Box(
            low=0, high=self.num_actions - 1, shape=(2,), dtype=int
        )
        self.reward_range = spaces.Box(low=0, high=np.inf, shape=(2,), dtype=float)

        self.end_state = torch.ones(2, dtype=int) * self.size
        self.start_state = torch.zeros(2, dtype=int)

        self.reset()

    def _is_end_state(self, state):
        return is_end_state(state_index=state, env=self).item()

    def _action_map(self, action, state):
        return self.action_map(state_index=state, action_index=action)

    def _r_cost(self, state, next_state):
        return self.r_cost(
            env=self, state_index=self.state, next_state_index=next_state
        ).item()

    def get_state_index(self) -> tuple:
        return self.state


class DiscreteReparamReverseEnv(DiscreteReparamEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_map, self.num_actions = get_action_map(
            self.depth, size=self.size, reverse=True
        )
        self.is_start_state = is_end_state
        self.is_end_state = is_start_state
        self.r_cost = r_cost_reverse
        self.start_state_index = torch.LongTensor([self.size - 1, self.size - 1])


@torch.no_grad()
def r_cost(
    state_index: torch.LongTensor, next_state_index: torch.LongTensor, env: ReparamEnv
) -> torch.Tensor:
    assert torch.all(next_state_index >= state_index), "wrong direction"
    if state_index[0] == next_state_index[0]:
        return zero_
    # elif torch.any(state_index == next_state_index):
    #     return penalty_

    _, q_data, r_data = env.t_data, env.q_data, env.r_data
    q_eval = q_data[state_index[0] : next_state_index[0] + 1]
    r_eval = r_data[state_index[1] : next_state_index[1] + 1]

    index_diff = next_state_index - state_index
    if state_index[1] == next_state_index[1]:
        # if x does not change compute
        tx_indices = torch.arange(0, index_diff[0] + 1)
        integrand = torch.sum(q_eval**2, dim=-1)

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
    state_index: torch.LongTensor, next_state_index: torch.LongTensor, env: ReparamEnv
) -> torch.Tensor:
    return r_cost(state_index=next_state_index, next_state_index=state_index, env=env)


def get_epsilon_greedy(epsilon: float, num_actions: int) -> callable:
    def epsilon_greedy(
        state_index: torch.LongTensor, model: callable, env: ReparamEnv
    ) -> torch.LongTensor:
        if torch.rand(1) < epsilon:
            return torch.randint(low=0, high=num_actions, size=(1,)).long()
        else:
            return get_optimal_action(q_model=model, state_index=state_index, env=env)

    return epsilon_greedy


def get_state(state_index: torch.LongTensor, env: ReparamEnv):
    """works also with multiple states"""
    t_data = env.data[:][0]
    return t_data[state_index].float()


def is_end_state(state_index: torch.LongTensor, env: ReparamEnv):
    t_data = env.data[:][0]
    grid_max_index = t_data.size(0) - 1
    answer = torch.logical_and(
        state_index[..., 0:1] == grid_max_index, state_index[..., 1:2] == grid_max_index
    )
    return answer


def is_start_state(state_index: torch.LongTensor, env: ReparamEnv):
    return torch.logical_and(state_index[..., 0:1] == 0, state_index[..., 1:2] == 0)


def get_optimal_action(
    q_model: callable, state_index: torch.LongTensor, env: ReparamEnv
) -> torch.LongTensor:
    state = get_state(state_index=state_index, env=env).float()
    action_index = torch.argmin(q_model(state), dim=-1, keepdim=True).long()
    return action_index


def get_action_map(depth, size: int, reverse: bool = False):
    # assumes same base
    # hack to make a list of all admissible directions (in reverse)
    max_grid_index = size - 1
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
        action_index: np.ndarray, state_index: np.ndarray
    ) -> torch.LongTensor:
        """state is index (i, j), action is (i)

        Works with many actions"""

        # torch.LongTensor(np.divmod(action_index, base_dim)[-1])

        # It is strange that actions act on state indices, but it should work
        action = action_array[action_index[..., 0]]
        new_state_index = torch.clip(
            state_index + action, min=0, max=max_grid_index
        ).long()
        return new_state_index

    return action_map, len(action_array)


def get_optimal_path(
    q_model: callable,
    env: ReparamEnv,
) -> Union[torch.LongTensor, None]:
    # q_model: callable, state_index: torch.LongTensor, data: dataset
    # no beautiful recursion here, python is stupid :(
    state_index = env.start_state_index.detach().clone()
    t_data = env.data[:][0]
    size, dim = t_data.size(0), len(state_index.flatten())
    index_tensor = torch.zeros((2 * size, dim), dtype=torch.long)
    index_tensor[0] = state_index
    for i in range(1, 2 * size):

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


def get_path_value(path: torch.LongTensor, env: ReparamEnv) -> torch.Tensor:
    if path is None:
        return torch.tensor(torch.inf)
    value = torch.zeros(1)
    for i in range(len(path) - 1):
        value += env.r_cost(state_index=path[i], next_state_index=path[i + 1], env=env)
    return value


def compute_loss_rl(model: callable, env: ReparamEnv):
    path = get_optimal_path(q_model=model, env=env)
    return get_path_value(path=path, env=env)


def sample_states(env: ReparamEnv, N: int = 1) -> State:
    grid_len = len(env.data)
    states = torch.randint(0, grid_len, size=(N, 1), dtype=torch.long).long()
    return torch.cat((states, states), dim=1)


def sample_action(
    env: ReparamEnv,
    N: int = 1,
) -> torch.LongTensor:
    actions = torch.randint(0, env.num_actions, size=(N, 1), dtype=torch.long).long()
    return actions


@torch.no_grad()
def plot_solution_rl(model, env: ReparamEnv, **kwargs):
    x_eval, *_ = env.data[:]
    size = len(x_eval)
    ind = torch.as_tensor(np.indices((size, size)).T)

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
