import torch.nn as nn


# Hyperparameters configuration (linear regression has zero)
def build_fnn(num_neurons):
    # Input "layer" is implicitly the input
    model = nn.Sequential(
        nn.Linear(1, num_neurons),  # Hidden layer
        nn.Tanh(),  # Activation
        nn.Linear(num_neurons, 1)  # Output layer
    )

    # MSE = Mean Squared Error
    loss_func = nn.MSELoss()
    return model, loss_func