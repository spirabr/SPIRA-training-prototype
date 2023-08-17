import torch
from torch.utils.data import *


# Train and test data split
def data_split(simple_1d_regression):
    return DataLoader(simple_1d_regression, shuffle=True)
