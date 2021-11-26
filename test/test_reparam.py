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
import test.test_curves as tc

# q = sqrt(abs(dc_dt)) *c
########
PATH_FIGURES = "../figures/"
########

SET_NAME = "13_adam_tanh_noq"
FOLDS = 1
N = 128  # training points

MODEL_PARAMS = {
    "input_dimension": [1],
    "output_dimension": [1],
    "n_hidden_layers": [2],
    "neurons": [30],
    "activation": ["tanh"],
}
TRAINING_PARAMS = {
    "model": [ResNET],
    "num_epochs": [1],
    "batch_size": [N + 20],
    "regularization_param": [1e-7],
    "optimizer": ["strong_wolfe"],
    "learning_rate": [0.1],
    "compute_loss": [compute_loss_reparam],
    "loss_func": [get_elastic_metric_loss(tc.q, constrain_cost=1e3)],
}

if __name__ == "__main__":
    # Data frame with data

    # Load data
    x_zeros = torch.zeros(10, 1, requires_grad=True)
    x_line = torch.linspace(1, 0, N, requires_grad=True).unsqueeze(1)
    x_ones = torch.ones(10, 1, requires_grad=True)
    x_train = torch.cat((x_zeros, x_line, x_ones))
    y_train = tc.r(x_train.detach())
    x_val = torch.linspace(0, 1, N, requires_grad=True).unsqueeze(1)
    y_val = tc.r(x_val.detach())
    data = TensorDataset(x_train, y_train)
    data_val = TensorDataset(x_val, y_val)

    model_params_iter = create_subdictionary_iterator(MODEL_PARAMS)
    training_params_iter = create_subdictionary_iterator(TRAINING_PARAMS)

    cv_results = k_fold_cv_grid(
        Model=ResNET,
        model_param_iter=model_params_iter,
        fit=fit_FFNN,
        training_param_iter=training_params_iter,
        data=data,
        val_data=data_val,
        folds=FOLDS,
        verbose=False,
        get_error=get_elastic_error_func(q=tc.q),
    )

    # plotting
    x_train_ = x_train.detach()
    x_sorted, indices = torch.sort(x_train_, dim=0)
    plot_kwargs = {
        "x_train": x_sorted,
        "model_compare": tc.ksi_example,
        "x_axis": "t",
        "y_axis": "ksi(t)",
    }
    plot_result(
        path_figures=PATH_FIGURES,
        plot_name=SET_NAME,
        **cv_results,
        plot_function=plot_reparametrization,
        function_kwargs=plot_kwargs,
    )
