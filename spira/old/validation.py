import torch
from torch.utils.data import *
import numpy as np
def run_epoch_test(model, data_loader, loss_func, device="cpu"):
    running_loss = []
    y_true = []
    y_pred = []
    for inputs, labels in data_loader:
        inputs = moveTo(inputs, device)
        labels = moveTo(labels, device)

        y_hat = model(inputs)

        loss = loss_func(y_hat, labels)

        # Now we are just grabbing some information we would like to have
        running_loss.append(loss.item())

        if isinstance(labels, torch.Tensor):
            # moving labels & predictions back to CPU for computing / storing predictions
            labels = labels.detach().cpu().numpy()
            y_hat = y_hat.detach().cpu().numpy()
            # add to predictions so far
            y_true.extend(labels.tolist())
            y_pred.extend(y_hat.tolist())


    y_pred = np.asarray(y_pred)

    if len(y_pred.shape) == 2 and y_pred.shape[1] > 1:
        # We have a classification problem, convert to labels
        y_pred = np.argmax(y_pred, axis=1)

    return y_pred



# Validate the model predictions
def validate(model, loss_func, test_dataset):
    # with torch.no_grad():
    #     X_features = torch.tensor(X_test, dtype=torch.float32)
    #     Y_pred = model(X_features).cpu().numpy()
    # return Y_pred
    testing_loader = DataLoader(test_dataset)
    model = model.eval()
    with torch.no_grad():
        Y_pred = run_epoch_test(model, testing_loader, loss_func)
    return
