import math
from abc import ABC, abstractmethod
from typing import Optional

from spira.core.domain.checkpoint import (
    Checkpoint,
    CheckpointBuilder,
    CheckpointManager,
    Step,
)
from spira.core.domain.dataloader import DataLoader
from spira.core.domain.loss import Loss, MultipleLossCalculator, Validation
from spira.core.domain.model import (
    LabelsBatch,
    Model,
    PredictionsBatch,
)
from spira.core.domain.optimizer import Optimizer
from spira.core.domain.scheduler import Scheduler


class ModelTrainer(ABC):
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        scheduler: Scheduler,
        train_loss_calculator: MultipleLossCalculator,
        test_loss_calculator: MultipleLossCalculator,
        checkpoint_builder: CheckpointBuilder,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss_calculator = train_loss_calculator
        self.test_loss_calculator = test_loss_calculator
        self.checkpoint_builder = checkpoint_builder

    @abstractmethod
    def train(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader,
        num_epochs: int,
        checkpoint: Checkpoint,
    ) -> Model:
        pass


class BasicModelTrainer(ModelTrainer):
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        scheduler: Scheduler,
        train_loss_calculator: MultipleLossCalculator,
        test_loss_calculator: MultipleLossCalculator,
        checkpoint_builder: CheckpointBuilder,
    ):
        super().__init__(
            model,
            optimizer,
            scheduler,
            train_loss_calculator,
            test_loss_calculator,
            checkpoint_builder,
        )

    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        previous_checkpoint: Optional[Checkpoint],
    ) -> Model:
        if num_epochs <= 0:
            raise ValueError("Number of epochs should be greater than zero")

        checkpoint_manager = CheckpointManager(
            self.checkpoint_builder,
            self.model,
            self.optimizer,
            previous_checkpoint,
        )

        for epoch in range(num_epochs):
            for features, labels in train_loader:
                predictions = self.model.predict(features)
                train_loss = self.train_loss_calculator.calculate(
                    self.to_validations(predictions, labels)
                )

                self.optimizer.zero_grad()
                self.train_loss_calculator.recalculate_weights()
                self.optimizer.step()
                self.scheduler.step()

                if self._has_loss_exploded(train_loss):
                    break

            idx = 0
            for features, labels in test_loader:
                step = Step(epoch * len(test_loader) + idx)

                predictions = self.model.predict(features)
                test_loss = self.test_loss_calculator.calculate(
                    self.to_validations(predictions, labels)
                )

                checkpoint_manager.update_and_save_checkpoints(test_loss, step)
                idx += 1

        return self.model

    @staticmethod
    def to_validations(predictions: PredictionsBatch, labels: LabelsBatch):
        return [
            Validation(prediction, label)
            for prediction, label in zip(predictions, labels)
        ]

    @staticmethod
    def _has_loss_exploded(current_loss: Loss) -> bool:
        current_loss_value = current_loss.item()
        return current_loss_value > 1e8 or math.isnan(current_loss_value)
