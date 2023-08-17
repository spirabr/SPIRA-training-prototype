import numpy as np


def data_creation():
    # Create a 1-dimensional input
    X = np.linspace(0, 20, num=200)
    # Create an output
    y = X + np.sin(X) * 2 + np.random.normal(size=X.shape)
    return X, y

