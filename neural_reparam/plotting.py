import matplotlib.pyplot as plt
import numpy as np
from deepthermal.plotting import plot_result_sorted
from neural_reparam.reparam_env import ReparamEnv
from neural_reparam.reinforcement_learning import get_optimal_path
import torch


def plot_reparametrization(model, x_train, model_compare, **kwargs):
    y_pred = model(x_train).detach()
    y_train = model_compare(x_train).detach()
    plot_result_sorted(
        x_train=x_train, x_pred=x_train, y_train=y_train, y_pred=y_pred, **kwargs
    )


def plot_curve(*curves, name=None, N=128):
    m = 10
    n = N * m
    interval = torch.linspace(0, 1, n).reshape(n, 1)
    for q in curves:
        Q = q(interval)
        plt.plot(Q[:, 0], Q[:, 1])
        plt.plot(Q[::m, 0], Q[::m, 1], "*")
    if name is not None:
        plt.savefig(name)
    plt.show()


def plot_curve_1d(*curves):
    N = 500
    interval = torch.linspace(0, 1, N).reshape(N, 1)
    for i, q in enumerate(curves):
        Q = q(interval)
        plt.plot(interval, Q, label=i)
    plt.grid()
    plt.legend()
    plt.show()


def plot_models_performance(
    models,
    data,
    loss,
):
    models_loss = np.zeros(len(models))

    for model in models:
        for model_k in model:
            models_loss += loss(model_k, data)
        models_loss /= len(model)


@torch.no_grad()
def plot_solution_rl(model, env: ReparamEnv, **kwargs):
    (x_eval,) = env.t_data
    size = len(x_eval)
    ind = torch.as_tensor(np.indices((size, size)).T)

    grid = x_eval[ind]

    # comptue cost
    cost_matrix = torch.min(model(grid).detach(), dim=-1)[0]
    computed_path_indexes = get_optimal_path(model=model, env=env)
    computed_path = x_eval[computed_path_indexes]

    # add V values to axes
    fig, ax = plt.subplots(1)
    plot = ax.imshow(cost_matrix, extent=[0, 1, 0, 1], origin="lower")
    fig.colorbar(plot)

    plot_result_sorted(
        x_pred=computed_path[:, 0], y_pred=computed_path[:, 1], fig=fig, **kwargs
    )
    return fig
