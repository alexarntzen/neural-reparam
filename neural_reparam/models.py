import torch.nn as nn


# GLOBAL VARIABLES

activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}


class CNN(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        neurons,
        kernel_size=5,
        activation="relu",
        **kwargs
    ):
        super(CNN, self).__init__()

        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        self.kernel_size = kernel_size + kernel_size % 2 - 1
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = activation

        self.activation_ = activations[self.activation]
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [
                nn.Conv1d(
                    1, 1, kernel_size=self.kernel_size, padding=(self.kernel_size) // 2
                )
                for _ in range(n_hidden_layers - 1)
            ]
        )
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

    def forward(self, x):
        # print("in", x.shape)
        if len(x.shape) == 1:
            x = x.reshape((1, 1) + x.shape)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        # The forward function performs the set of affine and
        # non-linear transformations defining the network
        # (see equation above)
        x = self.activation_(self.input_layer(x))

        # try using blocks
        for layer in self.hidden_layers:
            x = self.activation_(layer) + x

            # print(x.shape)
        x = x.squeeze()
        # print(x.shape)
        # print("out", self.output_layer(x).shape)
        return self.output_layer(x)

    def __str__(self):
        return "CNN"


class BResCNN(CNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        # print("in", x.shape)
        if len(x.shape) == 1:
            x = x.reshape((1, 1) + x.shape)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        # The forward function performs the set of affine and
        # non-linear transformations defining the network
        # (see equation above)
        x = self.activation_(self.input_layer(x))

        # try using blocks
        for first_layer, second_layer in pairs(self.hidden_layers):
            z = self.activation_(first_layer(x))
            x = self.activation_(second_layer(z)) + x

            # print(x.shape)
        x = x.squeeze()
        # print(x.shape)
        # print("out", self.output_layer(x).shape)
        return self.output_layer(x)

    def __str__(self):
        return "BResCNN"


class ResCNN(CNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x):
        # print("in", x.shape)
        if len(x.shape) == 1:
            x = x.reshape((1, 1) + x.shape)
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)
        # The forward function performs the set of affine and
        # non-linear transformations defining the network
        # (see equation above)
        x = self.activation_(self.input_layer(x))

        # try using blocks
        for layer in self.hidden_layers:
            x = self.activation_(layer) + x

            # print(x.shape)
        x = x.squeeze()
        # print(x.shape)
        # print("out", self.output_layer(x).shape)
        return self.output_layer(x)

    def __str__(self):
        return "ResCNN"


def init_zero(model, **kwargs):
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            if min(m.weight.shape) == 1:
                nn.init.eye_(m.weight)
            else:
                nn.init.zeros_(m.weight)
            m.bias.data.fill_(0)

    model.apply(init_weights)
    return model


def pairs(items):
    for i in range(len(items) // 2):
        yield items[2 * i], items[2 * i + 1]
