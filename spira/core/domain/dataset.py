import torch
from sklearn.model_selection import train_test_split

from spira.core.domain.audio import Audio


class Dataset(torch.utils.data.Dataset):
    def __init__(self, features: list[Audio], labels: list[int]):
        self.features = features
        self.labels = labels

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self.features):
            raise IndexError("Index out of bounds")
        return self.features[index], self.labels[index]


def create_train_and_test_datasets(
    features: list[Audio], labels: list[int], random_state: int
) -> tuple[Dataset, Dataset]:
    X_train, X_test, y_train, y_test = train_test_split(
        X=features, y=labels, train_size=0.6, test_size=0.4, random_state=random_state
    )
    return Dataset(X_train, y_train), Dataset(X_test, y_test)
