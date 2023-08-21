import torch
from torch.utils.data import *


# Feature engineering
def extract_features(X, y):
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    dataset = SimpleDataset(X, y)
    return dataset

class SimpleDataset(Dataset):

    def __init__(self, X, y):
        super(SimpleDataset, self).__init__()
        self.X = X.reshape(-1, 1)
        self.y = y.reshape(-1, 1)

    def __getitem__(self, index):
        inputs = torch.tensor(self.X[index, :], dtype=torch.float32)
        labels = torch.tensor(self.y[index], dtype=torch.float32)
        return inputs, labels

    def inputs(self):
        return torch.tensor(self.X, dtype=torch.float32)

    def labels(self):
        return torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]
