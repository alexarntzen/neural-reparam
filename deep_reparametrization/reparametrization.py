import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import DataLoader

l2_loss = nn.MSELoss()


def compute_loss_reparam(loss_func, model: callable, x_train, y_train):
    loss = loss_func(model, x_train, y_train)
    return loss


def get_elastic_metric_loss(r: callable, constrain_cost, verbose=False):
    """assumes we want to compute the values of q_1 beforehand
    q is the original curve"""
    ReLU = nn.ReLU()

    zero = torch.tensor([0.0])
    one = torch.tensor([1.0])

    def elastic_metric_loss(ksi_model: callable, x_train, y_train):
        q_eval = y_train
        dim = y_train.shape[1]
        ksi_eval = ksi_model(x_train)
        # each ksi_i is only dependent on x_i
        # need to sum the final ksi_prd because of batching
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


# new function used for logging. Takes in data = (x_eval, q_eval)
def get_elastic_error_func(r: callable, true_dist=0):
    def get_elastic_error(model, data):
        x_data, q_eval = next(
            iter(DataLoader(data, batch_size=len(data), shuffle=False))
        )
        ksi_eval = model(x_data)

        # each ksi_i is only dependent on x_i
        # need to sum the final ksi_prd because of batching
        ksi_dx = autograd.grad(ksi_eval.sum(), x_data)[0]

        # would not need abs here but sometimes it goes negative
        r_trans = torch.sqrt(torch.abs(ksi_dx)) * r(ksi_eval)
        return 2 * l2_loss(q_eval, r_trans) - true_dist

    return get_elastic_error
