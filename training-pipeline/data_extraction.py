import torch

from torch.utils.data import Dataset
from sklearn.datasets import fetch_openml


class SimpleDataset(Dataset):
    def __init__(self, X, y):
        super(SimpleDataset, self).__init__()
        self.X = X
        self.y = y

    def __getitem__(self, index: int):
        inputs = torch.tensor(self.X.iloc[index, :], dtype=torch.float32)
        targets = torch.tensor(int(self.y.iloc[index]), dtype=torch.int64)
        return inputs, targets

    def __len__(self):
        return self.X.shape[0]


def data_reading():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    print(X.shape)
    return SimpleDataset(X, y)