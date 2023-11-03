from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from spira.adapter.config import Config
from spira.core.domain.mish import Mish


# Template method
class Model(ABC):
    def __init__(self, config: Config):
        self.config = config
        self.num_features = config.num_features()
        self.mish = Mish()
        self.conv = self._build_cnn()
        self.fc1 = self._build_fc1(self.config, self.conv)
        self.fc2 = self._build_fc2(self.config)
        self.dropout = self._define_dropout()

    def forward(self, x):
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_feature]
        x = self.conv(x)
        # x: [B, n_filters, T, num_feature]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, n_filters, num_feature]
        x = self._reshape_x(x)
        # x: [B, T, fc2_dim]
        x = self.fc1(x)
        x = self.mish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        return x

    def _build_cnn(self):
        convs = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(7, 1), dilation=(2, 1)),
            nn.GroupNorm(16, 32),
            self.mish,
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(p=0.7),
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(2, 1)),
            nn.GroupNorm(8, 16),
            self.mish,
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(p=0.7),
            # cnn3
            nn.Conv2d(16, 8, kernel_size=(3, 1), dilation=(2, 1)),
            nn.GroupNorm(4, 8),
            self.mish,
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(p=0.7),
            # cnn4
            nn.Conv2d(8, 4, kernel_size=(2, 1), dilation=(1, 1)),
            nn.GroupNorm(2, 4),
            self.mish,
            nn.Dropout(p=0.7),
        ]
        return nn.Sequential(*convs)

    @abstractmethod
    def _build_fc1(self, config, conv) -> nn.Linear:
        pass

    def _build_fc2(self, config: Config) -> nn.Linear:
        return nn.Linear(config.model.fc1_dim, config.model.fc2_dim)

    def _define_dropout(self) -> nn.Dropout:
        return nn.Dropout(p=0.7)

    @abstractmethod
    def _reshape_x(self, x):
        pass


class MaxLengthPaddingModel(Model):
    def _build_fc1(self, config: Config, conv: torch.nn.modules.container.Sequential) -> nn.Linear:
        # it's very useful because if you change the convolutional architecture the model calculate its, and you don't need change this :)
        # I prefer activate the network in toy example because is easier than calculate the conv output
        # get zeros input
        inp = torch.zeros(1, 1, config.dataset.max_seq_len, config.num_features())
        # get out shape
        toy_activation_shape = self.conv(inp).shape
        # set fully connected input dim
        fc1_input_dim = (
            toy_activation_shape[1] * toy_activation_shape[2] * toy_activation_shape[3]
        )
        return nn.Linear(fc1_input_dim, config.model.fc1_dim)

    def _reshape_x(self, x):
        # x: [B, T*n_filters*num_feature]
        return x.view(x.size(0), -1)


class NoMaxLengthPaddingModel(Model):
    def _build_fc1(self, config: Config, conv: torch.nn.modules.container.Sequential) -> nn.Linear:
        # dynamic calculation num_feature, it's useful if you use max-pooling or other pooling in feature dim, and this model don't break
        inp = torch.zeros(1, 1, 500, self.num_features)
        # get out shape
        return nn.Linear(4 * self.conv(inp).shape[-1], config.model.fc1_dim)

    def _reshape_x(self, x):
        # x: [B, T, n_filters*num_feature]
        return x.view(x.size(0), x.size(1), -1)


def build_spira_model(config: Config) -> Model:
    if config.dataset.padding_with_max_length:
        return MaxLengthPaddingModel(config)
    return NoMaxLengthPaddingModel(config)
