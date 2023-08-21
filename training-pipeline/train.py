import torch
from torch.utils.data import *


def moveTo(obj, device):
    """
    obj: the python object to move to a device, or to move its contents to a device
    device: the compute device to move objects to
    """
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, list):
        return [moveTo(x, device) for x in obj]
    elif isinstance(obj, tuple):
        return tuple(moveTo(list(obj), device))
    elif isinstance(obj, set):
        return set(moveTo(list(obj), device))
    elif isinstance(obj, dict):
        to_ret = dict()
        for key, value in obj.items():
            to_ret[moveTo(key, device)] = moveTo(value, device)
        return to_ret
    else:
        return obj


def train_neural_network(model, loss_func, train_dataset, epochs=20, device="cpu"):
    training_loader = DataLoader(train_dataset)

    # WE create the optimizer and move the model to the compute device
    # SGD is Stochastic Gradient Decent over the parameters $\Theta$
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Place the model on the correct compute resource (CPU or GPU)
    model.to(device)

    # The next two for loops handle the Red steps, iterating through all the data (batches) multiple times (epochs)
    for epoch in range(epochs):

        # Put our model in training mode
        model = model.train()
        running_loss = 0.0

        for inputs, labels in training_loader:
            # Move the batch of data to the device we are using. this is the last red step
            inputs = moveTo(inputs, device)
            labels = moveTo(labels, device)

            # First a yellow step, prepare the optimizer. Most PyTorch code will do this first to make sure everything is in a clean and ready state.
            # Otherwise, it will have old information from a previous iteration
            optimizer.zero_grad()

            # The next two lines of code perform the two blue steps
            y_hat = model(inputs)  # this just computed $f_\theta(\boldsymbol{x_i})$

            # Compute loss
            loss = loss_func(y_hat, labels)

            # Now the remaining two yellow steps, compute the gradient and ".step()" the optimizer
            loss.backward()  # $\nabla_\Theta$ just got computed by this one call

            # Now we just need to update all the parameters
            optimizer.step()  # $\Theta_{k+1} = \Theta_k − \eta \cdot \nabla_\Theta \ell(\hat{y}, y)$

            # Now we are just grabbing some information we would like to have
            running_loss += loss.item()


