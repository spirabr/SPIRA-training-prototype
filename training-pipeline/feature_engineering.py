import numpy as np


# Feature engineering
def extract_features(X):
    return np.array(X).reshape(-1, 1)
