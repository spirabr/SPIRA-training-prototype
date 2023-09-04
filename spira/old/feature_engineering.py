import pandas as pd
import torch
from torch import tensor
from torch.utils.data import TensorDataset


# Feature engineering

def transform_inputs(X: pd.DataFrame) -> tensor:
    return torch.tensor(X, dtype=torch.float32)


def transform_label(y: pd.DataFrame) -> tensor:
    return torch.tensor(y, dtype=torch.long)


def build_training_dataset(X: tensor, y: tensor) -> TensorDataset:
    return TensorDataset(X, y)
