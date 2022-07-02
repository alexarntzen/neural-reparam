import matplotlib.pyplot as plt
import numpy as np
from extratorch.plotting import plot_result_sorted
from neural_reparam.reparam_env import DiscreteReparamEnv
from neural_reparam.reinforcement_learning import get_optimal_path
from typing import List, Union
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
def plot_solution_rl(
    model: Union[List[torch.nn.Module], torch.nn.Module],
    env: DiscreteReparamEnv,
    **kwargs
):
    fig, ax = plt.subplots(1, tight_layout=True)
    # get values
    x = t = env.t_data
    grid = np.stack(np.meshgrid(x, t, indexing="ij"), axis=-1)

    # compute cost
    model_cost_matrix = np.max(
        model(torch.as_tensor(grid, dtype=torch.float32)).numpy(), axis=-1
    )
    model_path = get_optimal_path(model=model, env=env, double_search=True)
    # will save and close fig
    fig = plot_value_func(
        value_matrix=model_cost_matrix,
        path=model_path,
        t_data=env.t_data,
        ax=ax,
        **kwargs
    )

    return fig


def plot_value_func(
    value_matrix: np.ndarray,
    t_data: np.ndarray,
    path: Union[List, np.ndarray] = None,
    ax: plt = None,
    colorbar=True,
    **kwargs
):
    if ax is None:
        fig, ax = plt.subplots(1, tight_layout=True)
    else:
        fig = ax.get_figure()

    # get grid
    x = t = t_data
    X, T = np.meshgrid(x, t, indexing="ij")

    cost_matrix = -value_matrix if np.any(value_matrix < 0) else value_matrix
    plot = ax.contourf(X, T, cost_matrix, cmap="viridis")

    if colorbar:
        fig.colorbar(plot, ax=ax, pad=0.02)
    if path is not None:
        # add path
        t_indexes = [p[0] for p in path]
        x_indexes = [p[1] for p in path]
        path_coordinates = t_data[t_indexes], t_data[x_indexes]
        plot_result_sorted(
            x_pred=path_coordinates[0],
            y_pred=path_coordinates[1],
            color_pred="red",
            ax=ax,
            pred_label="Optimal path",
            **kwargs
        )
    return fig
