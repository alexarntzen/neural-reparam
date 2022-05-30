"""These curves are the same mod DIff + """
import torch
from neural_reparam.plotting import plot_curve
from neural_reparam.interpolation import get_pc_curve, get_pl_curve
from neural_reparam.utils import stack_last_dim, use_torch_with_numpy
from math import pi


def c_2(t):
    c_x = torch.cos(2 * pi * t)
    c_y = torch.sin(4 * pi * t)
    return stack_last_dim(c_x, c_y)


@use_torch_with_numpy
def r(t):
    # sqrt(abs(c_2'))c_2
    r_x = -2 * pi * torch.sin(2 * pi * t)
    r_y = 4 * pi * torch.cos(4 * pi * t)
    return stack_last_dim(r_x, r_y)


const1 = 2 * torch.log(torch.tensor([21]))
const2 = 4 * torch.tanh(torch.tensor([10]))


@use_torch_with_numpy
def ksi(t):
    ksi1 = torch.log(20 * t + 1) / const1
    ksi2 = (1 + torch.tanh(20 * t - 10)) / const2
    return ksi2 + ksi1


def d_ksi_dt(t):
    if len(t.shape) == 1:
        t = torch.unsqueeze(t, 1)
    ksi1_dot = (20 / (20 * t + 1)) / const1
    ksi2_dot = (20 / torch.cosh(20 * t - 10) ** 2) / const2
    return ksi1_dot + ksi2_dot


def c_1(t):
    # c_1 = c_2 o ksi
    return c_2(ksi(t))


@use_torch_with_numpy
def q(t):
    # Q(c_1) = Q(c_2 o ksi) = sqrt(d/dt ksi) Q(c_2) o ksi
    return torch.sqrt(d_ksi_dt(t)) * r(ksi(t))


# run this to whow the curves
if __name__ == "__main__":
    # Data frame with dat
    plot_curve(c_1, name="../figures/curve_1/curve_c_1.pdf")
    plot_curve(c_2, name="../figures/curve_1/curve_c_2.pdf")
    plot_curve(q, name="../figures/curve_1/curve_q.pdf")
    plot_curve(r, name="../figures/curve_1/curve_r.pdf")
    plot_curve(get_pc_curve(q, 128), name="../figures/curve_1_pc/curve_q_pc.pdf")
    plot_curve(get_pc_curve(r, 128), name="../figures/curve_1_pc/curve_r_pc.pdf")
    plot_curve(get_pl_curve(q, 128), name="../figures/curve_1_pl/curve_q_pl.pdf")
    plot_curve(get_pl_curve(r, 128), name="../figures/curve_1_pl/curve_r_pl.pdf")
    # plot_curve_1d(ksi)
