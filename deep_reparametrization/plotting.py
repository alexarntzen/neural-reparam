import matplotlib.pyplot as plt
import numpy as np
from deepthermal.plotting import plot_result_sorted
import torch


def plot_reparametrization(model, x_train, model_compare, **kwargs):
    y_pred = model(x_train).detach()
    y_train = model_compare(x_train).detach()
    plot_result_sorted(
        x_train=x_train, x_pred=x_train, y_train=y_train, y_pred=y_pred, **kwargs
    )


def plot_curve(*curves):
    N = 500
    interval = torch.linspace(0, 1, N).reshape(N, 1)
    for q in curves:
        Q = q(interval)
        plt.plot(Q[:, 0], Q[:, 1])
        plt.plot(Q[::10, 0], Q[::10, 1], "*")
    plt.show()


def plot_curve_1d(*curves):
    N = 500
    interval = torch.linspace(0.1, 0.9, N).reshape(N, 1)
    for i, q in enumerate(curves):
        Q = q(interval)
        plt.plot(interval, Q, label=i)
        plt.plot(Q, interval, label=i)
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
