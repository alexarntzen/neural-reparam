"""
Cost functions associated with the SRV form. || q - sqrt(ksi_dx)r circ ksi||_{L^2}
"""
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import torch.autograd as autograd
from typing import Union, List

l2_loss = nn.MSELoss()


def get_elastic_metric_loss(r: callable, constrain_cost=0, verbose=False):
    """assumes we want to compute the values of q_1 beforehand
    q is the original curve"""
    ReLU = nn.ReLU()

    zero = torch.zeros((1, 1), dtype=torch.float)
    one = torch.ones((1, 1), dtype=torch.float)

    def elastic_metric_loss(ksi_model: callable, x_train, y_train):
        q_eval = y_train
        dim = y_train.shape[1]
        ksi_eval = ksi_model(x_train)
        # each ksi_i is only dependent on x_i
        # need to sum the final ksi_pred because of batching
        ksi_dx = autograd.grad(ksi_eval.sum(), x_train, create_graph=True)[0]
        # would not need abs here but sometimes it goes negative
        r_trans = torch.sqrt(torch.abs(ksi_dx)) * r(ksi_eval)
        loss = dim * l2_loss(q_eval, r_trans)
        # penalizes to enforce constraints
        boundary_penalties = ksi_model(zero) ** 2 + (ksi_model(one) - one) ** 2
        diff_penalty = torch.sum(ReLU(-ksi_dx)) * constrain_cost
        final_loss = loss + diff_penalty + boundary_penalties * constrain_cost

        if verbose:
            print(
                loss.item(),
                diff_penalty.item(),
                boundary_penalties.item() * constrain_cost,
            )
        return final_loss

    return elastic_metric_loss


def compute_loss_reparam(
    model: nn.Module, data: Union[List, TensorDataset], loss_func: callable
):
    x_train, y_train = data[:]
    loss = loss_func(model, x_train, y_train)
    return loss
