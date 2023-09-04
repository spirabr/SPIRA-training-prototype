import torch.nn as nn


# Hyperparameters configuration (linear regression has zero)
def build_fnn(num_neurons):
    # Input "layer" is implicitly the input
    model = nn.Sequential(
        nn.Linear(2, num_neurons),  # Hidden layer 1
        nn.Tanh(),  # Activation
        nn.Linear(num_neurons, num_neurons),  # Hidden layer 2
        nn.Tanh(),  # Activation
        nn.Linear(num_neurons, 2)  # Output layer
    )

    loss_func = nn.CrossEntropyLoss()
    return model, loss_func
