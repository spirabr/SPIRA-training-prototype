import torch
from torch.utils.data import *


# Configuração de hiperparâmetros (linear regression tem zero)
class Simple1DRegressionDataset(Dataset):

    def __init__(self, X, y):
        super(Simple1DRegressionDataset, self).__init__()
        self.X = X.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

    def __getitem__(self, index):
        inputs = torch.tensor(self.X[index, :], dtype=torch.float32)
        labels = torch.tensor(self.y[index], dtype=torch.float32)
        return inputs, labels

    def __len__(self):
        return self.X.shape[0]
