import warnings

import numpy as np
from itertools import product
from math import gcd
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
import torch
import torch.nn as nn
from typing import Union

import gym
from gym import spaces
from typing import Optional

try:
    from signatureshape.so3.dynamic_distance import local_cost as dp_local_cost

    SIGNATURSHAPE_COST = True
except ImportError:
    warnings.warn("Failed to import local cost")
    SIGNATURSHAPE_COST = False
# from neural_reparam.reinforcement_learning import get_optimal_path


zero_ = torch.tensor(0.0)
l2_loss = nn.MSELoss()


class ReparamEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        q_func: callable = None,
        r_func: callable = None,
        data: tuple = None,
        size: int = None,
        scale: float = 1,
        action_penalty=0,
    ):
        """data = (t_data, q_data, r_data)"""
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        assert (
            q_func is not None and r_func is not None and size is not None
        ) or data is not None

        if data is not None:
            self.t_data, self.q_data, self.r_data = data[:]
            self.size = len(self.t_data)

        if q_func is not None and r_func is not None:
            self.q = q_func
            self.r = r_func
            self.size = size
            if data is None:
                self.t_data = np.linspace(0, 1, size)
                self.q_data = q_func(self.t_data)
                self.r_data = r_func(self.t_data)
        else:
            self.q = interp1d(
                y=self.q_data,
                x=self.t_data.flatten(),
                axis=0,
                assume_sorted=True,
            )
            self.r = interp1d(
                y=self.r_data,
                x=self.t_data.flatten(),
                axis=0,
                assume_sorted=True,
            )

        self.scale = scale
        self.action_penaly = action_penalty

        self.rgb_array_white = np.empty([self.size, self.size, 3], dtype=int)
        self.rgb_array_white.fill(255)

    def reset(
        self, seed=None, return_info=False, options=None
    ) -> Union[gym.core.ObsType, tuple[gym.core.ObsType, dict]]:
        # We need the following line to seed self.np_random
        if seed is None:
            seed = 0
        super().reset(seed=seed)
        self.rgb_array = np.copy(self.rgb_array_white)

        self.state = self.start_state
        observation = self.state
        return (observation, dict()) if return_info else observation

    def step(
        self, action: gym.core.ActType
    ) -> tuple[gym.core.ObsType, float, bool, dict]:
        action, penalty = self._validate_action(action)
        # update state
        new_state = self._action_map(action=action, state=self.state)

        # An episode is done if the agent has reached the target
        done = self._is_end_state(state=new_state)

        # reward is negative cost
        # reward is scaled so that Q has the right size
        reward = (
            -self.cost(state=self.state, next_state=new_state) * self.scale - penalty
        )
        observation = new_state
        self.state = new_state
        if isinstance(observation, torch.Tensor):
            return observation.numpy(), reward, done, dict()
        else:
            return observation, reward, done, dict()

    def test_step(self, action):
        action, penalty = self._validate_action(action)
        # update state
        new_state = self._action_map(action=action, state=self.state)

        # An episode is done if the agent has reached the target
        done = self._is_end_state(state=new_state)

        # reward is negative cost
        reward = -self.cost(state=self.state, next_state=new_state) - penalty
        observation = new_state

        if isinstance(observation, torch.Tensor):
            return observation.numpy(), reward, done, dict()
        else:
            return observation, reward, done, dict()

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            self.rgb_array[self.get_state_index] = 0
            return self.rgb_array
        else:
            super(ReparamEnv, self).render(mode=mode)  # just raise an exception

    def _validate_action(self, action):
        raise NotImplementedError

    def preprocess_state(self, state: np.ndarray = None) -> np.ndarray:
        if state is None:
            return self.state
        else:
            return state

    def _is_end_state(self, state):
        return np.array_equal(state, self.end_state)

    def cost(self, state: np.ndarray, next_state: np.ndarray) -> np.float32:
        raise NotImplementedError


class RealReparamEnv(ReparamEnv):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.end_state = np.ones(2, dtype=np.float32)
        self.start_state = np.zeros(2, dtype=np.float32)

        self.reset()

    def _action_map(self, action, state):
        # dx_dt = np.clip(action, -1, np.inf).item() + 1
        #
        # dt = np.random.rand()/self.size
        # dx = dx_dt * dt
        # action_step = np.array([dx, dt], dtype=np.float32)
        return np.clip(self.state + action, 0, 1)

    def _validate_action(self, action) -> tuple[gym.core.ActType, float]:
        # action_mod = np.clip(action,-1,np.inf)
        # penalty = (action_mod - action)*self.action_penaly
        action = action * 0.5 + 0.5
        max_diff = self.end_state - self.state
        action_diff = np.clip(action - max_diff, 0, np.inf)
        action_diff -= 100 * np.clip(action, -np.inf, 0)
        penalty = np.sum(action_diff) * self.action_penaly
        action_mod = np.clip(action, 0, max_diff)
        return action_mod, penalty

    def get_state_index(self) -> tuple:
        return round(self.state[0] * self.size), round(self.state[1] * self.size)

    def cost(self, state: np.ndarray, next_state: np.ndarray) -> np.float32:
        """Probably could be more efficient"""
        # extract values
        t_0, x_0 = state
        t_1, x_1 = next_state
        dt, dx = next_state - state
        assert dt >= 0 and dx >= 0
        if dt == 0:
            return 0  # penalty

        num_integration_points = round(max(dt, dx) * self.size) + 2
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
    def __init__(
        self,
        illegal_action_penalty: float = 0,
        depth: int = 4,
        use_dp_cost: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.depth = depth
        self.illegal_action_penalty = illegal_action_penalty
        self.action_map, self.num_actions = get_action_map(depth, size=self.size)
        self.r_cost = r_cost
        self.observation_space = spaces.Box(
            low=0, high=self.size - 1, shape=(2,), dtype=int
        )
        self.action_space = spaces.Box(
            low=0, high=self.num_actions - 1, shape=(1,), dtype=int
        )
        self.use_dp_cost = use_dp_cost

        self.end_state = np.ones(2, dtype=int) * self.size - 1
        self.start_state = np.zeros(2, dtype=int)

        self.reset()

    def _action_map(self, action, state):
        return self.action_map(state_index=state, action_index=action)

    def cost(self, state, next_state):
        if self.use_dp_cost:
            return self.dp_local_cost(state=state, next_state=next_state)
        else:
            return self.r_cost(
                env=self,
                state_index=state,
                next_state_index=next_state,
                illegal_action_penalty=self.illegal_action_penalty,
            ).item()

    def get_state_index(self) -> tuple:
        return self.state

    def preprocess_state(self, state: np.ndarray = None) -> np.ndarray:
        return self.t_data[state]

    def _validate_action(self, action):
        assert 0 <= action < self.num_actions
        return action, 0

    def dp_local_cost(self, state, next_state):
        if SIGNATURSHAPE_COST:
            return dp_local_cost(
                *state, *next_state, q0=self.q_data, q1=self.r_data, I=self.t_data
            )
        else:
            raise ImportError("No dp local cost")


class DiscreteReparamReverseEnv(DiscreteReparamEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_map, self.num_actions = get_action_map(
            self.depth, size=self.size, reverse=True
        )
        self.r_cost = r_cost_reverse

        self.end_state, self.start_state = self.start_state, self.end_state
        self.reset()

    def cost(self, state, next_state):
        return super().cost(state=next_state, next_state=state)


def r_cost(
    state_index: np.ndarray,
    next_state_index: np.ndarray,
    env: ReparamEnv,
    illegal_action_penalty: float = 0,
) -> float:
    assert np.all(np.greater_equal(next_state_index, state_index)), "wrong direction"
    index_diff = next_state_index - state_index
    if index_diff[0] == 0:
        print("illegal", next_state_index, state_index)
        return illegal_action_penalty * (index_diff[1])

    # elif torch.any(state_index == next_state_index):
    #     return penalty_

    t_data, q_data, r_data = env.t_data, env.q_data, env.r_data
    q_eval = q_data[state_index[0] : next_state_index[0] + 1]
    r_eval = r_data[state_index[1] : next_state_index[1] + 1]

    if state_index[1] == next_state_index[1]:
        # if x does not change compute
        tx_indices = np.arange(0, index_diff[0] + 1)
        integrand = np.sum(q_eval**2, axis=-1)
    else:
        gcd = np.gcd(index_diff[..., 0], index_diff[..., 1])

        # find the lowest iteger to represent the points on the interval using ints
        product_index = np.prod(index_diff, dtype=int) // gcd
        # one extra since we want the end state included
        common_length = product_index + 1

        # compute common indexes
        t_spacing = index_diff[1] // gcd
        x_spacing = index_diff[0] // gcd
        t_indices = np.arange(0, common_length, t_spacing.item(), dtype=int)
        x_indices = np.arange(0, common_length, x_spacing.item(), dtype=int)
        tx_indices = np.union1d(t_indices, x_indices)
        # compute integrand with interpolated values
        r_int = interp1d(x_indices, r_eval, axis=0, kind="linear", assume_sorted=True)(
            tx_indices
        )
        q_int = interp1d(t_indices, q_eval, axis=0, kind="linear", assume_sorted=True)(
            tx_indices
        )
        ksi_diff = index_diff[1] / index_diff[0]
        integrand = np.sum((q_int - np.sqrt(ksi_diff) * r_int) ** 2, axis=-1)
    start_t = t_data[state_index[0]]
    end_t = t_data[next_state_index[0]]

    # print("internal", integrand,tx_indices, start_t, end_t)
    # compute integral
    r_int = trapezoid(integrand, tx_indices) * (end_t - start_t) / tx_indices[-1]
    return r_int


def r_cost_reverse(
    state_index: torch.LongTensor,
    next_state_index: torch.LongTensor,
    env: ReparamEnv,
    illegal_action_penalty: float = 0,
) -> torch.Tensor:
    return r_cost(
        state_index=next_state_index,
        next_state_index=state_index,
        env=env,
        illegal_action_penalty=illegal_action_penalty,
    )


def get_real_state(state_index: torch.LongTensor, env: ReparamEnv):
    """works also with multiple states"""
    t_data = env.t_data
    return t_data[state_index].float()


def is_end_state(state_index: np.ndarray, env: ReparamEnv):
    grid_max_index = len(env.t_data) - 1
    answer = np.logical_and(
        state_index[..., 0:1] == grid_max_index, state_index[..., 1:2] == grid_max_index
    )
    return answer


def is_start_state(state_index: torch.LongTensor, env: ReparamEnv):
    return torch.logical_and(state_index[..., 0:1] == 0, state_index[..., 1:2] == 0)


def get_action_map(depth, size: int, reverse: bool = False):
    # assumes same base
    # hack to make a list of all admissible directions (in reverse)
    max_grid_index = size - 1
    assert depth <= max_grid_index, "action space larger than state space"
    action_array = np.array(
        [
            [x, y]
            for x, y in product(range(1, depth + 1), range(1, depth + 1))
            if gcd(x, y) == 1
        ]
    )
    if reverse:
        action_array *= -1
    max_incline_t = (
        -np.min(action_array[:, 0]) if reverse else np.max(action_array[:, 0])
    )
    max_incline_x = (
        -np.min(action_array[:, 1]) if reverse else np.max(action_array[:, 1])
    )
    end_state_index = (
        np.zeros(2, dtype=int)
        if reverse
        else np.array([max_grid_index, max_grid_index])
    )
    num_actions = len(action_array)
    # could also have used np.unravel_index

    def action_map(action_index: np.ndarray, state_index: np.ndarray) -> np.ndarray:
        """state is index (i, j), action is (i)"""

        # It is strange that actions act on state indices, but it should work
        action = action_array[action_index].flatten()
        new_state_index = np.clip(np.add(state_index, action), 0, max_grid_index)

        # if future illegal action
        difference = np.abs(end_state_index - new_state_index)

        if np.any(difference == 0):
            new_state_index = end_state_index
        elif (
            difference[1] / difference[0] > max_incline_x
            or difference[0] / difference[1] > max_incline_t
        ):
            return action_map((action_index + 1) % num_actions, state_index=state_index)

        return new_state_index

    return action_map, len(action_array)


def sample_states(env: ReparamEnv, N: int = 1):
    grid_len = len(env.data)
    states = torch.randint(0, grid_len, size=(N, 1), dtype=torch.long).long()
    return torch.cat((states, states), dim=1)


def sample_action(
    env: ReparamEnv,
    N: int = 1,
) -> torch.LongTensor:
    actions = torch.randint(0, env.num_actions, size=(N, 1), dtype=torch.long).long()
    return actions
