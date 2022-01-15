import torch.nn as nn

# GLOBAL VARIABLES


activations = {
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
}


class ResNet(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        n_hidden_layers,
        neurons,
        activation="relu",
        **kwargs
    ):
        super(ResNet, self).__init__()

        # Number of input dimensions n
        self.input_dimension = input_dimension
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = activation

        self.activation_ = activations[self.activation]

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)]
        )
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

    def forward(self, x):
        # The forward function performs the set of affine and
        # non-linear transformations defining the network
        # (see equation above)
        x = self.activation_(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_(layer(x)) + x
        return self.output_layer(x)

    def __str__(self):
        return "ResNet"


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
