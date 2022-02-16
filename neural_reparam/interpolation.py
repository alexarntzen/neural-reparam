import torch
from torch.nn.functional import relu


def get_pc_curve_from_data(data) -> callable:
    k = data.shape[0]

    def pc_curve(t):
        # remove outside interval
        t = relu(t)
        t -= relu(t - 1)
        indexes = (t * (k - 1)).type(torch.long).flatten()
        return torch.as_tensor(data[indexes])

    return pc_curve


def get_pc_curve(curve: callable, k) -> callable:
    h = 1 / (k - 1)

    def pc_curve(t):
        res = torch.remainder(t, h)
        t_n = t - res
        return curve(t_n)

    return pc_curve


def get_pl_curve(curve: callable, k) -> callable:
    h = 1 / (k - 1)

    def pc_curve(t):
        res = torch.remainder(t, h)
        t_n = t - res
        return (1 - res * (k - 1)) * curve(t_n) + res * (k - 1) * curve(t_n + h)

    return pc_curve


def get_pl_curve_from_data(data) -> callable:
    k = data.shape[0]
    data_ = torch.as_tensor(data)

    def pl_curve(t):
        # t = relu(t)
        # t -= relu(t - 1)
        res = torch.remainder(t, 1 / (k - 1))
        indexes = ((t) * (k - 1)).type(torch.long).flatten()

        indexes = torch.clamp(indexes, 0, k - 1)
        next_index = torch.clamp(indexes + 1, 0, k - 1)
        return data_[indexes] * (1 - res * (k - 1)) + data_[next_index] * res * (k - 1)

    return pl_curve
