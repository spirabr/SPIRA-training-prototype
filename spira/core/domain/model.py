import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from spira.core.domain.audio import Audios, get_wavs_from_audios
from spira.core.domain.checkpoint import (
    Checkpoint,
    CheckpointCreator,
    CheckpointManager,
    Step,
)
from spira.core.domain.cnn_builder import CNNBuilder
from spira.core.domain.dataloader import DataLoader
from spira.core.domain.loss import Label, MultipleLossCalculator, Prediction, Validation
from spira.core.domain.mish import Mish
from spira.core.domain.optimizer import Optimizer
from spira.core.domain.scheduler import Scheduler


class PyTorchModel(nn.Module):
    def __init__(self, cnn_builder: CNNBuilder):
        self.conv = self._build_cnn()
        self.mish = Mish()

        self.fc1 = cnn_builder.build_fc1(self.conv)
        self.fc2 = cnn_builder.build_fc2()
        self.dropout = cnn_builder.define_dropout()
        self.reshape_x = cnn_builder.reshape_x

    def forward(self, x: Audios):
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
            nn.Conv2d(
                32, 16, kernel_size=(5, 1), dilation=(2, 1)
            ),  # Camada convolucional
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
        return nn.Sequential(*convs)


class Model(ABC):
    @abstractmethod
    def predict(self, features: Audios):
        pass


class ModelTrainer(ABC):
    def __init__(
        self,
        cnn_builder: CNNBuilder,
        optimizer: Optimizer,
        scheduler: Scheduler,
        train_loss_calculator: MultipleLossCalculator,
        validation_loss_calculator: MultipleLossCalculator,
        checkpoint_creator: CheckpointCreator,
    ):
        self.model = PyTorchModel(cnn_builder)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss_calculator = train_loss_calculator
        self.validation_loss_calculator = validation_loss_calculator
        self.checkpoint_creator = checkpoint_creator

    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        num_epochs: int,
        checkpoint: Checkpoint,
    ) -> Model:
        pass


class BasicModel(Model):
    def __init__(self, model: PyTorchModel):
        self.model = model

    def predict(self, features: Audios):
        return self.model(get_wavs_from_audios(features))


class BasicModelTrainer(ModelTrainer):
    def __init__(
        self,
        cnn_builder: CNNBuilder,
        optimizer: Optimizer,
        scheduler: Scheduler,
        train_loss_calculator: MultipleLossCalculator,
        validation_loss_calculator: MultipleLossCalculator,
        checkpoint_creator: CheckpointCreator,
    ):
        super().__init__(
            cnn_builder,
            optimizer,
            scheduler,
            train_loss_calculator,
            validation_loss_calculator,
            checkpoint_creator,
        )

    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        num_epochs: int,
        previous_checkpoint: Optional[Checkpoint],
    ) -> BasicModel:
        if num_epochs <= 0:
            raise ValueError("Number of epochs should be greater than zero")

        checkpoint_manager = CheckpointManager(
            self.checkpoint_creator,
            self.model,
            self.optimizer,
            previous_checkpoint,
        )

        early_epochs = 0

        for epoch in range(num_epochs):
            for features_batch, label_batch in train_loader:
                prediction_batch = self.model(features_batch)
                train_loss = self.train_loss_calculator.calculate(
                    self.to_validations(prediction_batch, label_batch)
                )

                self.optimizer.zero_grad()
                self.train_loss_calculator.recalculate_weights()
                self.optimizer.step()
                self.scheduler.step()

                if self._has_loss_exploded(train_loss):
                    break

            for idx, (features_batch, label_batch) in enumerate(validation_loader):
                step = Step(epoch * len(validation_loader) + idx)
                prediction_batch = self.model(features_batch)
                validation_loss = self.validation_loss_calculator.calculate(self.to_validations(prediction_batch, label_batch))

                checkpoint_manager.update_and_save_checkpoints(validation_loss, step)

        return BasicModel(self.model)

    @staticmethod
    def to_validations(prediction_batch: list[Prediction], label_batch: list[Label]):
        return [Validation(prediction, label) for prediction, label in zip(prediction_batch, label_batch)]

    @staticmethod
    def _has_loss_exploded(current_loss: float) -> bool:
        return current_loss > 1e8 or math.isnan(current_loss)


class MixupModel(Model):
    pass
