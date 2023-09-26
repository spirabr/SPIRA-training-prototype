import os
from abc import ABC, abstractmethod
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch import stack
from torch.nn.utils.rnn import pad_sequence

# Todo: fazer enum
CONTROL_CLASS = 0
PATIENT_CLASS = 1

class Dataset(Dataset):
    """
    Class for load a train and test from dataset generate by import_librispeech.py and others
    """

    def __init__(self, c, ap, datasets: list[pd.DataFrame], classes: pd.Series, noise):
        assert (len(datasets) == len(classes), "Datasets and classes should have the same lengths")

        # set random seed
        random.seed(c['seed'])
        torch.manual_seed(c['seed'])
        torch.cuda.manual_seed(c['seed'])
        np.random.seed(c['seed'])

        self.c = c
        self.ap = ap

        self.tuples_dataset_and_class = zip(datasets, classes)
        self.datasets = datasets
        self.classes = classes

        self.noise = noise

        self.train = False
        self.eval = False
        self.test = False

        self.max_seq_len = max_seq_length_calculator(c, ap, self.c.dataset['padding_with_max_length'], self.train, self.datasets)

    def get_max_seq_length(self):
        return self.max_seq_len

    def _get_feature_and_target(self, wav, class_name):
        if self.c.dataset['split_wav_using_overlapping']:
            start_slice = 0
            features = []
            targets = []
            step = self.ap.sample_rate * self.c.dataset['step']
            for end_slice in range(self.ap.sample_rate * self.c.dataset['window_len'], wav.shape[1], step):
                spec = self.ap.get_feature_from_audio(wav[:, start_slice:end_slice]).transpose(1, 2)
                features.append(spec)
                targets.append(torch.FloatTensor([class_name]))
                start_slice += step
            if len(features) == 1:
                feature = self.ap.get_feature_from_audio(
                    wav[:, :self.ap.sample_rate * self.c.dataset['window_len']]).transpose(1, 2)
                target = torch.FloatTensor([class_name])
            elif len(features) < 1:
                raise RuntimeError(
                    "ERROR: Some sample in your dataset is less than {} seconds! Change the size of the overleapping window (CONFIG.dataset['window_len'])".format(
                        self.c.dataset['window_len']))
            else:
                feature = torch.cat(features, dim=0)
                target = torch.cat(targets, dim=0)
        else:
            # feature shape (Batch_size, n_features, timestamp)
            feature = self.ap.get_feature_from_audio(wav)
            # transpose for (Batch_size, timestamp, n_features)
            feature = feature.transpose(1, 2)
            # remove batch dim = (timestamp, n_features)
            feature = feature.reshape(feature.shape[1:])
            if self.c.dataset['padding_with_max_length']:
                # padding for max sequence
                zeros = torch.zeros(self.max_seq_len - feature.size(0), feature.size(1))
                # append zeros before features
                feature = torch.cat([feature, zeros], 0)
                target = torch.FloatTensor([class_name])
            else:
                target = torch.zeros(feature.shape[0], 1) + class_name
        return feature, target

    def __getitem__(self, idx):
        # TODO: Responder: As duas próximas linhas funcionam para objetos zip?
        #  Supondo que funcione, é preferível manter a primeira opção?
        wav = self.tuples_dataset_and_class[idx][0]
        class_name = self.tuples_dataset_and_class[idx][1]

        # wav = self.datasets[idx]
        # class_name = self.classes[idx]

        # its assume that noise file is bigger than wav file !!
        if self.c.data_augmentation['insert_noise']:
            self._insert_noise(wav, idx, class_name)

        return self._get_feature_and_target(wav, class_name)

    def __len__(self):
        return len(self.datasets)

    def _define_num_noise(self, class_name: int) -> int:
        if class_name == CONTROL_CLASS:
            return self.c.data_augmentation['num_noise_control']
        if class_name == PATIENT_CLASS:
            return self.c.data_augmentation['num_noise_patient']

    def _insert_noise(self, wav, idx, class_name):
        # Experiments do different things within the dataloader. So depending on the experiment we will have a different random state here in get item. To reproduce the tests always using the same noise we need to set the seed again, ensuring that all experiments see the same noise in the test !!
        if self.test:
            random.seed(
                self.c['seed'] * idx)  # multiply by idx, because if not we generate same some for all files !
            torch.manual_seed(self.c['seed'] * idx)
            torch.cuda.manual_seed(self.c['seed'] * idx)
            np.random.seed(self.c['seed'] * idx)

        for _ in range(self._define_num_noise(class_name)):
            # choose random noise file
            noise_wav = self.noise[random.randint(0, len(self.noise))][0]
            noise_wav_len = noise_wav.shape[1]
            wav_len = wav.shape[1]
            noise_start_slice = random.randint(0, noise_wav_len - (wav_len + 1))
            # sum two different noise
            noise_wav = noise_wav[:, noise_start_slice:noise_start_slice + wav_len]
            # get random max amp for noise
            max_amp = random.uniform(self.c.data_augmentation['noise_min_amp'],
                                     self.c.data_augmentation['noise_max_amp'])
            reduce_factor = max_amp / float(noise_wav.max().numpy())
            noise_wav = noise_wav * reduce_factor
            wav = wav + noise_wav


def max_seq_length_calculator(c, ap, padding_with_max_length, train_mode, datasets: list[pd.DataFrame]):
    if c.dataset['max_seq_len']:
        return c.dataset['max_seq_len']
    if not padding_with_max_length or not train_mode:
        return None # raise Exception("Properties unavailable")
    min_len, max_len = _find_min_max_in_wav(c.audio['hop_length'], datasets)

    print("The Max Time dim length is: {} (+- {} seconds)".format(max_len, (
            max_len * c.audio['hop_length']) / ap.sample_rate))
    print("The Min Time dim length is: {} (+- {} seconds)".format(min_len, (
            min_len * c.audio['hop_length']) / ap.sample_rate))
    return max_len


def _find_min_max_in_wav(hop_length, datasets: list[pd.DataFrame]):
    seq_lens = [_calculate_seq_len_for_wav(dataset, hop_length) for dataset in datasets]
    return min(seq_lens), max(seq_lens)


def _calculate_seq_len_for_wav(wav: pd.DataFrame, hop_length):
    return int((wav.shape[1] / hop_length) + 1)


def _load_train_dataset(c, ap, datasets: list[pd.DataFrame], classes: pd.Series, noise):
    return DataLoader(
        dataset=Dataset(c, ap, datasets, classes, noise),
        batch_size=c.train_config['batch_size'],
        shuffle=True,
        num_workers=c.train_config['num_workers'],
        collate_fn=own_collate_fn,
        pin_memory=True,
        drop_last=True,
        sampler=None)


def _load_eval_dataset(c, ap, datasets: list[pd.DataFrame], classes: pd.Series, noise):
    return DataLoader(
        dataset=Dataset(c, ap, datasets, classes, noise),
        collate_fn=own_collate_fn,
        batch_size=c.test_config['batch_size'],
        shuffle=False,
        num_workers=c.test_config['num_workers'])


def _load_test_dataset(c, ap, datasets: list[pd.DataFrame], classes: pd.Series, noise):
    return DataLoader(
        dataset=Dataset(c, ap, datasets, classes, noise),
        collate_fn=teste_collate_fn,
        batch_size=c.test_config['batch_size'],
        shuffle=False,
        num_workers=c.test_config['num_workers'])


def _load_inference_dataset(c, ap, datasets: list[pd.DataFrame], classes: pd.Series, noise):
    return DataLoader(
        dataset=Dataset(c, ap, datasets, classes, noise),
        collate_fn=teste_collate_fn,
        batch_size=c.test_config['batch_size'],
        shuffle=False,
        num_workers=c.test_config['num_workers'])


def load_dataset(mode, c, ap):
    if mode not in dataset_loader:
        raise Exception("Unknown mode")
    return dataset_loader[mode](c, ap)


dataset_loader = {
    "train": _load_train_dataset,
    "eval": _load_eval_dataset,
    "test": _load_test_dataset,
    "inf": _load_inference_dataset
}


def own_collate_fn(batch):
    features = []
    targets = []
    for feature, target in batch:
        features.append(feature)
        # print(target.shape)
        targets.append(target)

    if len(features[0].shape) == 3:  # if dim is 3, we have a many specs because we use a overlapping
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

    if len(features[0].shape) == 3:  # if dim is 3, we have a many specs because we use a overlapping
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
