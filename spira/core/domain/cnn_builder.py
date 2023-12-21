from abc import ABC, abstractmethod

import torch
from torch import nn as nn

from spira.adapter.config import Config, ModelConfig
from spira.core.domain.audio import Audios


class CNNBuilder(ABC):
    def __init__(self, config: ModelConfig, num_features: int):
        self.config = config
        self.num_features = num_features

    @abstractmethod
    def build_fc1(self, conv) -> nn.Linear:
        pass

    def build_fc2(self) -> nn.Linear:
        return nn.Linear(self.config.fc1_dim, self.config.fc2_dim)

    def define_dropout(self) -> nn.Dropout:
        return nn.Dropout(p=0.7)

    @abstractmethod
    def reshape_x(self, x):
        pass


class PaddedFeaturesCNNBuilder(CNNBuilder):
    def __init__(self, config: ModelConfig, num_features: int, max_audio_length: int):
        super().__init__(config, num_features)
        self.max_audio_length = max_audio_length

    def build_fc1(self, conv: torch.nn.modules.container.Sequential) -> nn.Linear:
        # it's very useful because if you change the convolutional architecture the model calculate its, and you don't need change this :)
        # I prefer activate the network in toy example because is easier than calculate the conv output
        # get zeros input
        inp = torch.zeros(1, 1, self.max_audio_length, self.num_features)
        # get out shape
        toy_activation_shape = conv(inp).shape
        # set fully connected input dim
        fc1_input_dim = (
            toy_activation_shape[1] * toy_activation_shape[2] * toy_activation_shape[3]
        )
        return nn.Linear(fc1_input_dim, self.config.fc1_dim)

    def reshape_x(self, x):
        # x: [B, T*n_filters*num_feature]
        return x.view(x.size(0), -1)


class OverlappedFeaturesCNNBuilder(CNNBuilder):
    def __init__(self, config: ModelConfig, num_features: int):
        super().__init__(config, num_features)

    def build_fc1(self, conv: torch.nn.modules.container.Sequential) -> nn.Linear:
        # dynamic calculation num_feature, it's useful if you use max-pooling or other pooling in feature dim, and this model don't break
        inp = torch.zeros(1, 1, 500, self.num_features)
        # get out shape
        return nn.Linear(4 * conv(inp).shape[-1], self.config.fc1_dim)

    def reshape_x(self, x):
        # x: [B, T, n_filters*num_feature]
        return x.view(x.size(0), x.size(1), -1)


def create_cnn_builder(config: Config, audios: Audios) -> CNNBuilder:
    if config.feature_engineering.split_wav_using_overlapping:
        return OverlappedFeaturesCNNBuilder(
            config.model, config.audio_processor.num_features()
        )
    if config.feature_engineering.padding_with_max_length:
        return PaddedFeaturesCNNBuilder(
            config.model,
            config.audio_processor.num_features(),
            audios.get_max_audio_length(),
        )
    raise ValueError("Unknown feature engineering type")
