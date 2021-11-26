import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader

l2_loss_ = nn.MSELoss()


def compute_loss_reparam(loss_func, model: callable, x_train, y_train):
    loss = loss_func(model, x_train, y_train)
    return loss


def get_elastic_metric_loss(q: callable, constrain_cost):
    """assumes we want to compute the values of q_1 beforehand
    q is the original curve"""
    ReLU = nn.ReLU()

    zero = torch.tensor([0.0])
    one = torch.tensor([1.0])

    def discrete_penalty_elastic(ksi_model: callable, x_train, y_train):
        r_eval = y_train
        ksi_eval = ksi_model(x_train)
        # each ksi_i is only dependent on x_i
        # need to sum the final ksi_prd because of batching
        ksi_dx = autograd.grad(ksi_eval.sum(), x_train, create_graph=True)[0]

        c_eval = q(ksi_eval).detach()
        # would not need abs here but sometimes it goes negative
        q_eval = torch.sqrt(torch.abs(ksi_dx)) * c_eval
        loss = l2_loss_(r_eval, q_eval)
        # penalizes to enforce constraints
        boundary_penalties = ksi_model(zero) ** 2 + (ksi_model(one) - one) ** 2
        diff_penalty = torch.sum(ReLU(-ksi_dx + 1e-14)) * constrain_cost
        print(loss.item(), diff_penalty.item(), boundary_penalties.item())
        return loss + diff_penalty + boundary_penalties * constrain_cost

    return discrete_penalty_elastic


# new function used for loggng
def get_elastic_error_func(q: callable):
    def get_elastic_error(model, data, type_str="", verbose=False):
        x_data, y_data = next(
            iter(DataLoader(data, batch_size=len(data), shuffle=False))
        )
        r_eval = y_data
        ksi_eval = model(x_data)

        # each ksi_i is only dependent on x_i
        # need to sum the final ksi_prd because of batching
        ksi_dx = autograd.grad(ksi_eval.sum(), x_data)[0]

        # would not need abs here but sometimes it goes negative
        q_eval = torch.sqrt(torch.abs(ksi_dx)) * q(ksi_eval)
        return l2_loss_(r_eval, q_eval)

    return get_elastic_error
