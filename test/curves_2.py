"""These curves are not the same mod Diff+"""
import torch
from math import pi, sqrt
from deep_reparametrization.plotting import plot_curve


# parametrization
def ksi(t):
    return t - torch.sin(2 * pi * t) / (2 * pi)


def d_ksi_dt(t):
    return 1 - torch.cos(2 * pi * t)


# cure 1
def c_1(t):
    return pi ** (-1 / 3) * torch.cat((torch.cos(pi * t), torch.sin(pi * t)), dim=-1)


def q(t):
    q_x = torch.cos(pi * t)
    q_y = torch.sin(pi * t)
    return torch.cat((q_x, q_y), dim=-1)


# curve 2 reparameterized
def c_2(t):
    # c_1 = c_2 o ksi
    c_x = torch.zeros_like(t)
    c_y = torch.pow(3 * t + 1, 1 / 3)
    return torch.cat((c_x, c_y), dim=-1)


def r(t):
    # r = Q(c_2)
    r_x = torch.zeros_like(t)
    r_y = torch.ones_like(t)
    return torch.cat((r_x, r_y), dim=-1)


DIST_R_Q = 2 - sqrt(2)

# run this to show the curves
if __name__ == "__main__":
    # Data frame with dat
    plot_curve(c_1)
    plot_curve(c_2)
    plot_curve(q)
    plot_curve(r)
