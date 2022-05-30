"""These curves are not the same mod Diff+"""
import torch
from math import pi, sqrt
from neural_reparam.plotting import plot_curve, plot_curve_1d
from neural_reparam.utils import stack_last_dim, use_torch_with_numpy


# parametrization
@use_torch_with_numpy
def ksi(t):
    return t - torch.sin(2 * pi * t) / (2 * pi)


@use_torch_with_numpy
def d_ksi_dt(t):
    return 1 - torch.cos(2 * pi * t)


# cure 1
@use_torch_with_numpy
def c_1(t):
    return pi ** (-1 / 3) * stack_last_dim(torch.cos(pi * t), torch.sin(pi * t))


@use_torch_with_numpy
def q(t):
    q_x = torch.cos(pi * t)
    q_y = torch.sin(pi * t)
    return stack_last_dim(q_x, q_y)


# curve 2 reparameterized
@use_torch_with_numpy
def c_2(t):
    # c_1 = c_2 o ksi
    c_x = torch.zeros_like(t)
    c_y = torch.pow(3 * t + 1, 1 / 3)
    return stack_last_dim(c_x, c_y)


@use_torch_with_numpy
def r(t):
    # r = Q(c_2)
    r_x = torch.zeros_like(t)
    r_y = torch.ones_like(t)
    return stack_last_dim(r_x, r_y)


DIST_R_Q = 2 - sqrt(2)

# run this to show the curves
if __name__ == "__main__":
    # Data frame with dat
    plot_curve(c_1, name="../figures/curve_2/curve_c_1.pdf")
    plot_curve(c_2, name="../figures/curve_2/curve_c_2.pdf")
    plot_curve(q, name="../figures/curve_2/curve_q.pdf")
    plot_curve(r, name="../figures/curve_2/curve_r.pdf")
    plot_curve_1d(ksi)
