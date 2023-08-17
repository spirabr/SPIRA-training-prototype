import torch

# Predição
def validate(model, X):
    device = torch.device("cpu")
    with torch.no_grad():
        Y_pred = model(torch.tensor(X.reshape(-1, 1), device=device, dtype=torch.float32)).cpu().numpy()
    return Y_pred
