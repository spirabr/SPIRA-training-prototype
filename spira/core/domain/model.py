from typing import NewType

import torch
import torch.nn as nn

from spira.adapter.config import TrainingOptionsConfig
from spira.core.domain.cnn_builder import CNNBuilder
from spira.core.domain.mish import Mish

FeaturesBatch = NewType("FeaturesBatch", torch.Tensor)
LabelsBatch = NewType("LabelsBatch", torch.Tensor)
PredictionsBatch = NewType("PredictionsBatch", torch.Tensor)

Parameter = NewType("Parameter", torch.nn.Parameter)


class TorchModel(nn.Module):
    def __init__(self, cnn_builder: CNNBuilder):
        super().__init__()

        self.conv = self._build_cnn()
        self.mish = Mish()

        self.fc1 = cnn_builder.build_fc1(self.conv)
        self.fc2 = cnn_builder.build_fc2()
        self.dropout = cnn_builder.define_dropout()
        self.reshape_x = cnn_builder.reshape_x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, num_feature]
        x = x.unsqueeze(1)
        # x: [B, 1, T, num_feature]
        x = self.conv(x)
        # x: [B, n_filters, T, num_feature]
        x = x.transpose(1, 2).contiguous()
        # x: [B, T, n_filters, num_feature]
        x = self.reshape_x(x)
        # x: [B, T, fc2_dim]
        x = self.fc1(x)
        x = self.mish(x)
        x = self.dropout(x)
        x = self.fc2(x)
        y = torch.sigmoid(x)
        return y

    def _build_cnn(self) -> nn.Sequential:
        layers = [
            # cnn1
            nn.Conv2d(1, 32, kernel_size=(7, 1), dilation=(2, 1)),
            nn.GroupNorm(16, 32),
            self.mish,
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Dropout(p=0.7),
            # cnn2
            nn.Conv2d(32, 16, kernel_size=(5, 1), dilation=(2, 1)),
            nn.GroupNorm(8, 16),  # Normalizacao
            self.mish,  # suavizacao da camada anterior
            nn.MaxPool2d(kernel_size=(2, 1)),  # pooling
            nn.Dropout(p=0.7),  # activation function
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
        return nn.Sequential(*layers)


class Model:
    def __init__(self, model: TorchModel):
        self.model = model

    def predict(self, features: FeaturesBatch) -> PredictionsBatch:
        return self.model(torch.Tensor(features))

    def dump_state(self) -> dict:
        return self.model.state_dict()

    def load_state(self, state_dict: dict):
        self.model.load_state_dict(state_dict)

    def get_parameters(self) -> list[Parameter]:
        return [Parameter(parameter) for parameter in self.model.parameters()]


class BasicModel(Model):
    def __init__(self, model: TorchModel):
        super().__init__(model)


def _create_torch_model(cnn_builder: CNNBuilder) -> TorchModel:
    return TorchModel(cnn_builder)


def create_model(_options: TrainingOptionsConfig, cnn_builder: CNNBuilder) -> Model:
    return BasicModel(_create_torch_model(cnn_builder))
