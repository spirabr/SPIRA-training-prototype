import torch


# Validate the model predictions
def validate(model, test_features):
    with torch.no_grad():
        Y_pred = model(test_features.inputs()).cpu().numpy()
    return Y_pred
