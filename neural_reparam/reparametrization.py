"""
Cost functions associated with the SRV form. || q - sqrt(ksi_dx)r circ ksi||_{L^2}
"""
import torch
import torch.nn as nn
import torch.autograd as autograd

l2_loss = nn.MSELoss()


def get_elastic_metric_loss(r: callable, constrain_cost=0, verbose=False):
    """assumes we want to compute the values of q_1 beforehand
    q is the original curve"""
    ReLU = nn.ReLU()

    zero = torch.zeros((1, 1), dtype=torch.long)
    one = torch.ones((1, 1), dtype=torch.long)

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


def compute_loss_reparam(loss_func, model: callable, x_train, y_train):
    loss = loss_func(model, x_train, y_train)
    return loss


def batch_reinforcement_learning(Q: callable, r: callable, states, next):
    """On policy Q learning with discretized action and time
    Q(s): (X x T) --> [0, inf )^#A
    """
    # choose actions and compute next
    Q_est = Q(states)
    actions = torch.amax(Q_est).no_grad()
    states_next = next(states, actions)

    r_est = r(states, actions)

    # calculate max
    Q_est_next = Q(states_next).no_grad() * torch.unsqueeze(states == 1, dim=-1)

    # calculate goal
    Y_est = r_est + Q_est_next
    return l2_loss(Y_est - Q_est)
