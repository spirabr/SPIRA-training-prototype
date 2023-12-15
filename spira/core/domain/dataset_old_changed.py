import pandas as pd
import torch
from torch import stack
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from spira.core.domain.audio import Audio


class Dataset(Dataset):
    """
    Class for load a train and test from dataset generate by import_librispeech.py and others
    """

    def __init__(self, c, ap, audios: list[Audio], classes: pd.Series):
        self.c = c
        self.ap = ap

        self.wav_with_class_list = zip(audios, classes)
        self.audios = audios
        self.classes = classes

        self.max_seq_len = self._calculate_max_seq_length()

    def _calculate_max_seq_length(self):
        if self.c.dataset["max_seq_len"]:
            return self.c.dataset["max_seq_len"]
        if not self.c.dataset["padding_with_max_length"] or not self.train:
            return None  # raise Exception("Properties unavailable")
        min_len, max_len = _find_min_max_in_wav(
            self.c.audio_processor["hop_length"], self.audios
        )

        print(
            "The Max Time dim length is: {} (+- {} seconds)".format(
                max_len,
                (max_len * self.c.audio_processor["hop_length"])
                / self.ap.preferred_sample_rate,
            )
        )
        print(
            "The Min Time dim length is: {} (+- {} seconds)".format(
                min_len,
                (min_len * self.c.audio_processor["hop_length"])
                / self.ap.preferred_sample_rate,
            )
        )
        return max_len

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        wav, class_name = self.wav_with_class_list[idx]
        return self._get_feature_and_target(wav, class_name)

    def _get_feature_and_target(self, wav: Audio, class_name):
        if self.c.dataset["split_wav_using_overlapping"]:
            return self._get_feature_and_target_using_overlapping(wav, class_name)

        # TODO: Should I do a function for the next three commands?

        # feature shape (Batch_size, n_features, timestamp)
        feature = self.ap.transform_into_feature(wav)
        # transpose for (Batch_size, timestamp, n_features)
        feature = feature.transpose(1, 2)
        # remove batch dim = (timestamp, n_features)
        feature = feature.reshape(feature.shape[1:])

        # TODO: Should I do functions for the if-else clauses?

        if self.c.dataset["padding_with_max_length"]:
            # padding for max sequence
            zeros = torch.zeros(self.max_seq_len - feature.size(0), feature.size(1))
            # append zeros before features
            feature = torch.cat([feature, zeros], 0)
            target = torch.FloatTensor([class_name])
        else:
            target = torch.zeros(feature.shape[0], 1) + class_name

        return feature, target

    def _get_feature_and_target_using_overlapping(self, wav: Audio, class_name):
        features, targets = self._append_features_and_targets_using_window_len(
            wav, class_name
        )

        if len(features) == 1:
            feature = self.ap.transform_into_feature(
                wav[:, : self.ap.preferred_sample_rate * self.c.dataset["window_len"]]
            ).transpose(1, 2)
            target = torch.FloatTensor([class_name])
            return feature, target

        # TODO: How should I do that error check before? Or Is it even possible to get that error at this time?
        if len(features) < 1:
            raise RuntimeError(
                "ERROR: Some sample in your dataset is less than {} seconds! Change the size of the overleapping window (CONFIG.dataset['window_len'])".format(
                    self.c.dataset["window_len"]
                )
            )

        feature = torch.cat(features, dim=0)
        target = torch.cat(targets, dim=0)

        return feature, target

    def _append_features_and_targets_using_window_len(self, wav: Audio, class_name):
        start_slice = 0
        features = []
        targets = []
        step = self.ap.preferred_sample_rate * self.c.dataset["step"]
        for end_slice in range(
            self.ap.preferred_sample_rate * self.c.dataset["window_len"],
            wav.shape[1],
            step,
        ):
            spec = self.ap.transform_into_feature(
                wav[:, start_slice:end_slice]
            ).transpose(1, 2)
            features.append(spec)
            targets.append(torch.FloatTensor([class_name]))
            start_slice += step
        return features, targets

    def get_max_seq_length(self):
        return self.max_seq_len


def _find_min_max_in_wav(hop_length, datasets: list[pd.DataFrame]):
    seq_lens = [_calculate_seq_len_for_wav(dataset, hop_length) for dataset in datasets]
    return min(seq_lens), max(seq_lens)


def _calculate_seq_len_for_wav(wav: pd.DataFrame, hop_length):
    return int((wav.shape[1] / hop_length) + 1)


def _load_train_dataset(c, ap, datasets: list[pd.DataFrame], classes: pd.Series, noise):
    return DataLoader(
        dataset=Dataset(c, ap, datasets, classes, noise),
        batch_size=c.train_config["batch_size"],
        shuffle=True,
        num_workers=c.train_config["num_workers"],
        collate_fn=own_collate_fn,
        pin_memory=True,
        drop_last=True,
        sampler=None,
    )


def _load_eval_dataset(c, ap, datasets: list[pd.DataFrame], classes: pd.Series, noise):
    return DataLoader(
        dataset=Dataset(c, ap, datasets, classes, noise),
        collate_fn=own_collate_fn,
        batch_size=c.test_config["batch_size"],
        shuffle=False,
        num_workers=c.test_config["num_workers"],
    )


def _load_test_dataset(c, ap, datasets: list[pd.DataFrame], classes: pd.Series, noise):
    return DataLoader(
        dataset=Dataset(c, ap, datasets, classes, noise),
        collate_fn=teste_collate_fn,
        batch_size=c.test_config["batch_size"],
        shuffle=False,
        num_workers=c.test_config["num_workers"],
    )


def _load_inference_dataset(
    c, ap, datasets: list[pd.DataFrame], classes: pd.Series, noise
):
    return DataLoader(
        dataset=Dataset(c, ap, datasets, classes, noise),
        collate_fn=teste_collate_fn,
        batch_size=c.test_config["batch_size"],
        shuffle=False,
        num_workers=c.test_config["num_workers"],
    )


def load_dataset(mode, c, ap):
    if mode not in dataset_loader:
        raise Exception("Unknown mode")
    return dataset_loader[mode](c, ap)


dataset_loader = {
    "train": _load_train_dataset,
    "eval": _load_eval_dataset,
    "test": _load_test_dataset,
    "inf": _load_inference_dataset,
}


def own_collate_fn(batch):
    features = []
    targets = []
    for feature, target in batch:
        features.append(feature)
        # print(target.shape)
        targets.append(target)

    if (
        len(features[0].shape) == 3
    ):  # if dim is 3, we have a many specs because we use a overlapping
        targets = torch.cat(targets, dim=0)
        features = torch.cat(features, dim=0)
    else:
        # padding with zeros timestamp dim
        features = pad_sequence(features, batch_first=True, padding_value=0)
        # its padding with zeros but mybe its a problem because
        targets = pad_sequence(targets, batch_first=True, padding_value=0)

    #
    targets = targets.reshape(targets.size(0), -1)
    # list to tensor
    # targets = stack(targets, dim=0)
    # features = stack(features, dim=0)
    # print(features.shape, targets.shape)
    return features, targets


def teste_collate_fn(batch):
    features = []
    targets = []
    slices = []
    targets_org = []
    for feature, target in batch:
        features.append(feature)
        targets.append(target)
        if len(feature.shape) == 3:
            slices.append(torch.tensor(feature.shape[0]))
            # its used for integrity check during unpack overlapping for calculation loss and accuracy
            targets_org.append(target[0])

    if (
        len(features[0].shape) == 3
    ):  # if dim is 3, we have a many specs because we use a overlapping
        targets = torch.cat(targets, dim=0)
        features = torch.cat(features, dim=0)
    else:
        # padding with zeros timestamp dim
        features = pad_sequence(features, batch_first=True, padding_value=0)
        # its padding with zeros but maybe its a problem because
        targets = pad_sequence(targets, batch_first=True, padding_value=0)

    #
    targets = targets.reshape(targets.size(0), -1)

    if slices:
        slices = stack(slices, dim=0)
        targets_org = stack(targets_org, dim=0)
    else:
        slices = None
        targets_org = None
    # list to tensor
    # targets = stack(targets, dim=0)
    # features = stack(features, dim=0)
    # print(features.shape, targets.shape)
    return features, targets, slices, targets_org
