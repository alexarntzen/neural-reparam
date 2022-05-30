import unittest

import numpy as np
import torch
from gym.utils.env_checker import check_env
from neural_reparam.interpolation import get_pl_curve_from_data
from neural_reparam.reinforcement_learning import get_path_value
from neural_reparam.reparam_env import (
    r_cost,
    DiscreteReparamEnv,
    RealReparamEnv,
    DiscreteReparamReverseEnv,
)

try:
    from signatureshape.so3.dynamic_distance import local_cost

    LOCAL_COST_EXISTS = True
except ImportError:
    print("No such module")
    LOCAL_COST_EXISTS = False

import experiments.curves as c1

import timeit

from neural_reparam.reparametrization import get_elastic_metric_loss


class TestEnv(unittest.TestCase):
    def test_gym_reparm_env(self):
        # Load data
        for N in [256, 1024, 4096]:
            # data

            t_train = np.linspace(0, 1, N)
            r_train = c1.r(t_train)
            q_train = c1.q(t_train)
            data = (t_train, q_train, r_train)

            env = RealReparamEnv(r_func=c1.r, q_func=c1.q, size=N)
            check_env(env)
            env = DiscreteReparamEnv(r_func=c1.r, q_func=c1.q, size=N)
            check_env(env)
            env = DiscreteReparamReverseEnv(r_func=c1.r, q_func=c1.q, size=N)

            env = RealReparamEnv(data=data)
            check_env(env)
            env = DiscreteReparamEnv(data=data)
            check_env(env)
            env = DiscreteReparamReverseEnv(data=data)
            check_env(env)

    def test_r_integral(self):
        # Load data
        for _ in range(10):
            for N in [256, 1024, 4096]:
                # data
                t_train = np.linspace(0, 1, N)
                q_train = c1.q(t_train)
                r_train = c1.r(t_train)
                data = (t_train, q_train, r_train)
                env = DiscreteReparamEnv(data=data)
                # random index and diff
                start = np.random.randint(0, N // 3, size=2)
                diff = np.random.randint(0, N // 3, size=2)

                # construct indexes for torch
                middle = start + diff
                end = start + 2 * diff

                # sum of partials for same approx
                part_1 = r_cost(start, middle, env=env).item()
                part_2 = r_cost(middle, end, env=env).item()
                total = r_cost(start, end, env=env).item()
                self.assertAlmostEqual(part_1 + part_2, total, delta=1e-3)

                # positivity
                self.assertTrue(total >= 0)

    def test_r_integral_real(self):
        # Load data
        for _ in range(10):
            for N in [256, 1024, 4096]:
                # data
                env = RealReparamEnv(r_func=c1.r, q_func=c1.q, size=N)
                # random index and diff
                start = np.random.rand(2)
                diff = np.random.rand(2)

                # construct indexes for torch
                middle = start + diff
                end = start + 2 * diff

                # sum of partials for same approx
                part_1 = env._r_cost(start, middle)
                part_2 = env._r_cost(middle, end)
                total = env._r_cost(start, end)
                self.assertAlmostEqual(part_1 + part_2, total, delta=1e-3)

                # positivity
                self.assertTrue(total >= 0)

    def test_compare_local(self, check_time=False):
        # Load data
        print("\nComparing to local_cost function:")
        for _ in range(10):
            for N in [256, 1024, 4096]:
                if local_cost is None:
                    break
                # random index
                start = np.random.randint(0, N, size=2)
                end = np.random.randint(start, (N, N), size=2)
                if end[1] == start[1]:
                    continue

                # data
                x_train = np.linspace(0, 1, N)
                q_train = c1.q(x_train)
                r_train = c1.r(x_train)
                data = (x_train, q_train, r_train)
                env = DiscreteReparamEnv(data=data)
                env_real = RealReparamEnv(r_func=c1.r, q_func=c1.q, size=N)
                env_real2 = RealReparamEnv(data=data)
                N = env.size

                # compute result
                r_eval = r_cost(state_index=start, next_state_index=end, env=env).item()
                r_eval_real = env_real._r_cost(start / N, end / N)
                r_eval_real2 = env_real2._r_cost(start / N, end / N)
                if LOCAL_COST_EXISTS:
                    local_cost_eval = local_cost(
                        *start, *end, q0=q_train, q1=r_train, I=x_train
                    )

                    print(
                        f"r_real_cost:{r_eval_real}, r_cost:{r_eval},"
                        f" local_cost: {local_cost_eval}"
                    )
                    # compare to descrete
                    self.assertAlmostEqual(
                        r_eval, local_cost_eval, delta=local_cost_eval * 100 / N
                    )
                    # compare to real
                    self.assertAlmostEqual(
                        r_eval_real, local_cost_eval, delta=local_cost_eval * 100 / N
                    )

                # compare two reals
                self.assertAlmostEqual(
                    r_eval_real, r_eval_real2, delta=local_cost_eval * 100 / N
                )
                if check_time:

                    def test_1():
                        r_cost(
                            state_index=start,
                            next_state_index=end,
                            env=env,
                        ).item()

                    def test_2():
                        local_cost(*start, *end, q0=q_train, q1=r_train, I=x_train)

                    print(
                        "\n Time r_cost:",
                        timeit.timeit(test_1, number=10),
                        "local_cost:",
                        timeit.timeit(test_2, number=10),
                    )

    def test_compare_ksi(self):
        print("\nComputing estimated optimal solution:")
        # Load data
        for N in [2 ** (n * 2) for n in range(5, 8)]:
            # data
            x_train = np.linspace(0, 1, N)
            q_train = c1.q(x_train)
            r_train = c1.r(x_train)
            data = (x_train, q_train, r_train)
            env = DiscreteReparamEnv(data=data)

            # calculate r_cost
            path = get_solution_path(N)
            value = get_path_value(path=path, env=env)

            # data
            sol_curve = path[:, 1] / (N - 1)

            n = np.sqrt(N).astype(int)
            x_train = np.linspace(0, 1, n)
            q_train = c1.q(x_train)
            r_train = c1.r(x_train)
            # calculate with pl-estimation
            path_ksi = get_pl_curve_from_data(
                data=torch.unsqueeze(torch.as_tensor(sol_curve), 1)
            )
            penalty_free_loss_func = get_elastic_metric_loss(
                r=c1.r, constrain_cost=0, verbose=False
            )
            x_torch = torch.unsqueeze(torch.tensor(x_train, requires_grad=True), 1)

            pl_value = penalty_free_loss_func(
                path_ksi, x_torch, torch.as_tensor(q_train)
            ).item()

            print(f"r_cost: {value}, true cost: {pl_value}")
            self.assertAlmostEqual(pl_value, value, delta=100 / np.log(n))

            # print(
            #     "\n Time r_cost:",
            #     timeit.timeit(test_1, number=10),
            #     "local_cost:",
            #     timeit.timeit(test_2, number=10),
            # )


def get_solution_path(N):
    """gives the optimal solution for  grid N with sqrt(N) eqidistant times"""
    assert np.sqrt(N).is_integer(), "N is not sqare"
    n = int(np.sqrt(N))
    t_indexes = np.arange(1, N, n)
    t_values = np.linspace(0, 1, n)
    x_values = c1.ksi(t_values)
    x_indexes = np.round(x_values * (N - 1)).astype(int)
    return np.stack((t_indexes, x_indexes), axis=-1)


if __name__ == "__main__":
    unittest.main()
