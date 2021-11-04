import torch
from torch.utils.data import TensorDataset

from deepthermal.FFNN_model import FFNN, fit_FFNN, init_xavier
from deepthermal.validation import create_subdictionary_iterator, k_fold_cv_grid

from deepthermal.plotting import plot_result

from deep_reparametrization.plotting import plot_reparametrization
from deep_reparametrization.reparametrization import (
    get_elastic_metric_loss,
    compute_loss_reparam,
)
import test.test_curves as tc


# q = sqrt(abs(dc_dt)) *c
########
PATH_FIGURES = "../figures/"
########

SET_NAME = "new_q_6"
FOLDS = 1
N = 1000  # training points
constrain_cost = 100

MODEL_PARAMS = {
    "input_dimension": [1],
    "output_dimension": [1],
    "n_hidden_layers": [3],
    "neurons": [40],
    "activation": ["relu"],
}
TRAINING_PARAMS = {
    "num_epochs": [500],
    "batch_size": [265],
    "regularization_param": [1e-5],
    "optimizer": ["ADAM"],
    "learning_rate": [0.01],
    "compute_loss": [compute_loss_reparam],
    "loss_func": [get_elastic_metric_loss(tc.q_orig, constrain_cost)],
}

if __name__ == "__main__":
    # Data frame with data

    # Load data
    x_train = torch.rand((N, 1), requires_grad=True)
    y_train = tc.q_reparam(x_train.detach())
    data = TensorDataset(x_train, y_train)
    val_data = TensorDataset(x_train, y_train)
    model_params_iter = create_subdictionary_iterator(MODEL_PARAMS)
    training_params_iter = create_subdictionary_iterator(TRAINING_PARAMS)

    cv_results = k_fold_cv_grid(
        Model=FFNN,
        model_param_iter=model_params_iter,
        fit=fit_FFNN,
        training_param_iter=training_params_iter,
        data=data,
        val_data=val_data,
        init=init_xavier,
        folds=FOLDS,
        verbose=True,
    )

    # plotting
    x_train_ = x_train.detach()
    x_sorted, indices = torch.sort(x_train_, dim=0)
    plot_kwargs = {
        "x_train": x_sorted,
        "model_compare": tc.ksi_example,
        "x_axis": "t_in",
        "y_axis": "t_out",
    }
    plot_result(
        path_figures=PATH_FIGURES,
        plot_name=SET_NAME,
        **cv_results,
        plot_function=plot_reparametrization,
        function_kwargs=plot_kwargs,
    )
