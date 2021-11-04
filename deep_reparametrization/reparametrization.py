import torch
import torch.nn as nn
import torch.autograd as autograd

larning_rates = {"ADAM": 0.001, "LBFGS": 0.1}

activations = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), "tanh": nn.Tanh()}


def compute_loss_reparam(loss_func, model: callable, x_train, y_train):
    loss = loss_func(model, x_train, y_train)
    return loss


def get_elastic_metric_loss(q: callable, constrain_cost):
    """assumes we want to compute the values of q_1 beforehand
    q is the original curve"""
    l2_loss = nn.MSELoss()
    ReLU = nn.ReLU()
    constrain_cost = constrain_cost

    zero = torch.tensor([0.0])
    one = torch.tensor([1.0])

    def discrete_elastic(reparam_model: callable, x_train, y_train):
        q_1_eval = y_train
        ksi_pred = reparam_model(x_train)
        # each ksi_i is only dependent on x_i
        # need to sum the final ksi_prd because of batching
        ksi_dx = autograd.grad(ksi_pred.sum(), x_train, create_graph=True)[0]

        # would not need abs here but sometimes it goes negative
        q_2_reparam = torch.sqrt(torch.abs(ksi_dx)) * q(ksi_pred)
        loss = l2_loss(q_1_eval, q_2_reparam)

        # penalizes to enforce constraints
        start_penalties = torch.abs(reparam_model(zero)) + torch.abs(
            reparam_model(one) - one
        )
        diff_penalty = torch.sum(ReLU(-ksi_dx + 1e-8)) * constrain_cost
        return loss + diff_penalty + start_penalties * constrain_cost

    return discrete_elastic
