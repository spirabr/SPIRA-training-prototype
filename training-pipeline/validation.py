import torch
from torch.utils.data import *

# Validate the model predictions
def validate(model, X_test):
    with torch.no_grad():
        X_features = torch.tensor(X_test, dtype=torch.float32)
        Y_pred = model(X_features).cpu().numpy()
    return Y_pred
