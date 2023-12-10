import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


class SpiraDataset(Dataset):

    def __init__(
        self,
        data: dict
    ):
        self.inputs = data['inputs']
        self.labels = data['labels']
        self.max_seq_len = self._calculate_max_seq_length()


    def train_and_test_split_dataset(self):
        X = self.inputs
        y = self.labels
        hardcoded_random_state = 41
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, test_size=0.4, random_state=hardcoded_random_state)
        return X_train, X_test, y_train, y_test

    def _calculate_max_seq_length(self):
        if self.c.dataset["max_seq_len"]:
            return self.c.dataset["max_seq_len"]
        if not self.c.dataset["padding_with_max_length"] or not self.train:
            return None  # raise Exception("Properties unavailable")
        min_len, max_len = _find_min_max_in_wav(self.c.audio["hop_length"], self.wavs)

        print(
            "The Max Time dim length is: {} (+- {} seconds)".format(
                max_len, (max_len * self.c.audio["hop_length"]) / self.ap.sample_rate
            )
        )
        print(
            "The Min Time dim length is: {} (+- {} seconds)".format(
                min_len, (min_len * self.c.audio["hop_length"]) / self.ap.sample_rate
            )
        )
        return max_len


def _find_min_max_in_wav(hop_length, datasets: list[pd.DataFrame]):
    seq_lens = [_calculate_seq_len_for_wav(dataset, hop_length) for dataset in datasets]
    return min(seq_lens), max(seq_lens)


def _calculate_seq_len_for_wav(wav: pd.DataFrame, hop_length):
    return int((wav.shape[1] / hop_length) + 1)







