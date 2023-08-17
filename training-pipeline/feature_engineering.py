# Feature engineering
def feature_reshape(X, y):
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return X, y
