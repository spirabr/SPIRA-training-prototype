from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SpiraDataset(Dataset):
    def __init__(self, data: dict):
        self.inputs = data["inputs"]
        self.labels = data["labels"]

    def train_and_test_split_dataset(self):
        X = self.inputs
        y = self.labels
        hardcoded_random_state = 41
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.6, test_size=0.4, random_state=hardcoded_random_state
        )
        return X_train, X_test, y_train, y_test
