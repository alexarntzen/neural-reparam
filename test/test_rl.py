import unittest

import numpy as np
import torch
from torch.utils.data import TensorDataset

from neural_reparam.interpolation import get_pl_curve_from_data
from neural_reparam.reparam_env import r_cost, get_path_value, DiscreteReparamEnv
from so3.dynamic_distance import local_cost
import experiments.curves as c1

import timeit

from neural_reparam.reparametrization import get_elastic_metric_loss


class TestEnv(unittest.TestCase):
    def test_r_integral(self):
        # Load data
        for _ in range(10):
            for N in [256, 1024, 4096]:
                # data
                x_train = torch.linspace(0, 1, N, requires_grad=True)
                q_train = c1.q(x_train.unsqueeze(1).detach())
                r_train = c1.r(x_train.unsqueeze(1).detach())
                data = TensorDataset(x_train, q_train, r_train)
                env = DiscreteReparamEnv(data=data)
                # random index and diff
                start = np.random.randint(0, N // 3, size=2)
                diff = np.random.randint(0, N // 3, size=2)

                # construct indexes for torch
                middle = start + diff
                end = start + 2 * diff
                start_index = torch.LongTensor(start)
                middle_index = torch.LongTensor(middle)
                end_index = torch.LongTensor(end)

                # sum of partials for same approx
                part_1 = r_cost(start_index, middle_index, env=env).item()
                part_2 = r_cost(middle_index, end_index, env=env).item()
                total = r_cost(start_index, end_index, env=env).item()
                self.assertAlmostEqual(part_1 + part_2, total, delta=1e-3)

                # positivity
                self.assertTrue(total >= 0)

    def test_compare_local(self, check_time=False):
        # Load data
        print("\nComparing to local_cost function:")
        for _ in range(10):
            for N in [256, 1024, 4096]:

                # random index
                start = np.random.randint(0, N, size=2)
                end = np.random.randint(start, (N, N), size=2)
                if end[1] == start[1]:
                    continue
                start_index = torch.LongTensor(start)
                end_index = torch.LongTensor(end)

                # data
                x_train = torch.linspace(0, 1, N, requires_grad=False)
                q_train = c1.q(x_train.unsqueeze(1).detach())
                r_train = c1.r(x_train.unsqueeze(1).detach())
                data = TensorDataset(x_train, q_train, r_train)
                env = DiscreteReparamEnv(data=data)
                # compute result
                r_eval = r_cost(
                    state_index=start_index, next_state_index=end_index, env=env
                ).item()

                local_cost_eval = local_cost(
                    *start, *end, q0=q_train, q1=r_train, I=x_train.numpy()
                )
                print(f"r_cost:{r_eval}, local_cost: {local_cost_eval}")
                self.assertAlmostEqual(
                    r_eval, local_cost_eval, delta=local_cost_eval * 100 / N
                )

                if check_time:

                    def test_1():
                        r_cost(
                            state_index=start_index,
                            next_state_index=end_index,
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
        for N in [256**2, 512**2, 1024**2]:
            # data
            x_train = torch.linspace(0, 1, N, requires_grad=True)
            q_train = c1.q(x_train.unsqueeze(1).detach())
            r_train = c1.r(x_train.unsqueeze(1).detach())
            data = TensorDataset(x_train, q_train, r_train)
            env = DiscreteReparamEnv(data=data)

            # calculate r_cost
            path = get_solution_path(N)
            value = get_path_value(path=path, env=env).item()

            # data
            x_train = x_train.unsqueeze(1)
            sol_curve = path[:, 1] / (N - 1)
            sol_curve = sol_curve.requires_grad_(True).unsqueeze(1)

            # calculate with pl-estimation
            path_ksi = get_pl_curve_from_data(data=sol_curve)
            penalty_free_loss_func = get_elastic_metric_loss(
                r=c1.r, constrain_cost=0, verbose=False
            )
            pl_value = penalty_free_loss_func(path_ksi, x_train, q_train).item()

            print(f"r_cost: {value}, E: {pl_value}")
            self.assertAlmostEqual(pl_value, value, delta=1)

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
    t_indexes = torch.arange(1, N, n)
    t_values = torch.linspace(0, 1, n)
    x_values = c1.ksi(t_values)
    x_indexes = torch.round(x_values * (N - 1))

    return torch.stack((t_indexes, x_indexes), dim=-1).type(torch.LongTensor)


if __name__ == "__main__":
    unittest.main()
