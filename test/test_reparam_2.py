"""Test reparam of curves that are not equivqalent """
import torch
from torch.utils.data import TensorDataset
from deepthermal.FFNN_model import fit_FFNN
from deepthermal.validation import create_subdictionary_iterator, k_fold_cv_grid

from deepthermal.plotting import plot_result

from deep_reparametrization.plotting import plot_reparametrization
from deep_reparametrization.reparametrization import (
    get_elastic_metric_loss,
    compute_loss_reparam,
    get_elastic_error_func,
)
from deep_reparametrization.ResNET import ResNET
import test.curves_2 as tc2

# q = sqrt(abs(dc_dt)) *c
########
PATH_FIGURES = "../figures/curve_2"
########

SET_NAME = "13_curve_2"

FOLDS = 1
N = 2000  # training points internal
N_e = 40  # training points edge

MODEL_PARAMS = {
    "model": [ResNET],
    "input_dimension": [1],
    "output_dimension": [1],
    "n_hidden_layers": [2],
    "neurons": [5],
    "activation": ["tanh"],
}
TRAINING_PARAMS = {
    "num_epochs": [1],
    "batch_size": [N + 20],
    "regularization_param": [1e-8],
    "optimizer": ["strong_wolfe"],
    "learning_rate": [0.0001],
    "compute_loss": [compute_loss_reparam],
    "loss_func": [get_elastic_metric_loss(tc2.r, constrain_cost=1e4)],
}


if __name__ == "__main__":
    # Data frame with data

    # Load data
    x_zeros = torch.zeros(N_e // 2, 1, requires_grad=True)
    x_line = torch.linspace(1, 0, N, requires_grad=True).unsqueeze(1)
    x_ones = torch.ones(N_e // 2, 1, requires_grad=True)
    x_train = torch.cat((x_zeros, x_line, x_ones))
    y_train = tc2.q(x_train.detach())

    x_val = torch.linspace(0, 1, N, requires_grad=True).unsqueeze(1)
    y_val = tc2.q(x_val.detach())

    data = TensorDataset(x_train, y_train)
    data_val = TensorDataset(x_val, y_val)

    model_params_iter = create_subdictionary_iterator(MODEL_PARAMS)
    training_params_iter = create_subdictionary_iterator(TRAINING_PARAMS)

    cv_results = k_fold_cv_grid(
        model_params=model_params_iter,
        fit=fit_FFNN,
        training_params=training_params_iter,
        data=data,
        val_data=data_val,
        folds=FOLDS,
        verbose=True,
        get_error=get_elastic_error_func(r=tc2.r, true_dist=tc2.DIST_R_Q),
    )

    # plotting
    x_train_ = x_train.detach()
    x_sorted, indices = torch.sort(x_train_, dim=0)
    plot_kwargs = {
        "x_train": x_sorted,
        "model_compare": tc2.ksi,
        "x_axis": "t",
        "y_axis": "$\\xi(t)$",
    }
    plot_result(
        path_figures=PATH_FIGURES,
        plot_name=SET_NAME,
        **cv_results,
        plot_function=plot_reparametrization,
        function_kwargs=plot_kwargs,
    )
