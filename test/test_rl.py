import unittest
import random
import torch.autograd as autograd

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset

from neural_reparam.interpolation import get_pl_curve_from_data, get_pl_curve
from neural_reparam.reinforcement_learning import r_cost, get_state, get_path_value
from so3.dynamic_distance import local_cost
import experiments.curves as c1

import timeit

from neural_reparam.reparametrization import get_elastic_metric_loss


class TestEnv(unittest.TestCase):
    def test_r_cost(self):
        # Load data
        N = 32

        x_train = torch.linspace(0, 1, N, requires_grad=True)
        q_train = c1.q(x_train.unsqueeze(1).detach())
        r_train = c1.r(x_train.unsqueeze(1).detach())

        data = TensorDataset(x_train, q_train, r_train)

        start = torch.LongTensor([0, 0])
        middle = torch.LongTensor([8, 8])
        end = torch.LongTensor([16, 16])
        one = r_cost(start, middle, data).item()
        two = r_cost(middle, end, data).item()
        three = r_cost(start, end, data).item()
        print(one + two, three)
        self.assertAlmostEqual(one + two, three, delta=1e-3)

        start_index = torch.LongTensor([16, 28])
        end_index = torch.LongTensor([18, 31])
        cost = r_cost(
            state_index=start_index, next_state_index=end_index, data=data
        ).item()
        print(cost)
        self.assertTrue(cost >= 0)

        # linear value
        start_index = torch.LongTensor([0, 0])
        end_index = torch.LongTensor([N - 1, N - 1])
        cost = r_cost(
            state_index=start_index, next_state_index=end_index, data=data
        ).item()
        print(cost)

    def test_compare_local(self, check_time=True):
        # Load data
        print("\nComaring to local_cost function:")
        for _ in range(1):
            for N in [256, 1024, 4096]:

                start = np.random.randint(0, N, size=2)
                end = np.random.randint(start, (N, N), size=2)
                if end[1] == start[1]:
                    continue
                start_index = torch.LongTensor(start)
                end_index = torch.LongTensor(end)

                x_train = torch.linspace(0, 1, N, requires_grad=False)
                q_train = c1.q(x_train.unsqueeze(1).detach())
                r_train = c1.r(x_train.unsqueeze(1).detach())

                data = TensorDataset(x_train, q_train, r_train)
                r_eval = r_cost(
                    state_index=start_index, next_state_index=end_index, data=data
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
                        r_eval = r_cost(
                            state_index=start_index,
                            next_state_index=end_index,
                            data=data,
                        ).item()

                    def test_2():
                        local_cost_eval = local_cost(
                            *start, *end, q0=q_train, q1=r_train, I=x_train
                        )

                    print(
                        "\n Time r_cost:",
                        timeit.timeit(test_1, number=10),
                        "local_cost:",
                        timeit.timeit(test_2, number=10),
                    )

    def test_ksi(self):
        print("\nComputing estimated optimal solution:")
        # Load data
        for N in [2 ** 14]:
            x_train = torch.linspace(0, 1, N, requires_grad=True)
            q_train = c1.q(x_train.unsqueeze(1).detach())
            r_train = c1.r(x_train.unsqueeze(1).detach())
            data = TensorDataset(x_train, q_train, r_train)
            path = get_solution_path(N)
            value = get_path_value(path=path, data=data)
            print(f"E(ksi) value: {value.item()}")

            x_train = x_train.unsqueeze(1)
            sol_curve = path[:, 1] / (N - 1)
            sol_curve = sol_curve.requires_grad_(True).unsqueeze(1)

            penalty_free_loss_func = get_elastic_metric_loss(
                r=c1.r, constrain_cost=0, verbose=False
            )

            path_ksi = get_pl_curve_from_data(data=sol_curve)
            path_ksi_eval = path_ksi(x_train)
            path_ksi_dt = autograd.grad(
                path_ksi_eval.sum(), x_train, create_graph=True
            )[0]

            print("otp path cont :", penalty_free_loss_func(path_ksi, x_train, q_train))

            ksi_eval = c1.ksi(x_train)
            ksi_dt = autograd.grad(ksi_eval.sum(), x_train, create_graph=True)[0]

            pl_ksi = get_pl_curve(curve=c1.ksi, k=N)
            pl_ksi_eval = pl_ksi(x_train)
            pl_ksi_dt = autograd.grad(pl_ksi_eval.sum(), x_train, create_graph=True)[0]

            print("pl ksi cont :", penalty_free_loss_func(pl_ksi, x_train, q_train))
            print("abs:", torch.max(torch.abs((pl_ksi_eval - path_ksi_eval))))
            print("diff abs:", torch.max(torch.abs(path_ksi_dt - pl_ksi_dt)))
            # with torch.no_grad():
            #     plt.plot(path_ksi_dt, label="path")
            #     plt.plot(pl_ksi_dt, label="pl")
            #     plt.plot(ksi_dt, label="true")
            #     plt.legend()
            #     plt.show()


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
