from sklearn.datasets import make_moons

def generate_data():
    X, y = make_moons(n_samples=2000, noise=0.05)
    return X, y

