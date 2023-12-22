import math
import os
from typing import NewType, Optional, cast

import torch
from typing_extensions import Self

from spira.adapter.config import TrainCheckpointConfig
from spira.adapter.valid_path import ValidPath
from spira.core.domain.loss import Loss
from spira.core.domain.model import Model
from spira.core.domain.optimizer import Optimizer

Step = NewType("Step", int)


class Checkpoint:
    def __init__(self, checkpoint_state: dict):
        self.model_state = checkpoint_state["model"]
        self.optimizer_state = checkpoint_state["optimizer"]
        self.validation_loss = checkpoint_state["validation_loss"]
        self.step = Step(checkpoint_state["step"])

    @classmethod
    def create_initial_checkpoint(cls, model: Model, optimizer: Optimizer) -> Self:
        return cast(Self, Checkpoint.create(model, optimizer, Loss(math.inf), Step(0)))

    @classmethod
    def load(cls, checkpoint_path: ValidPath) -> Self:
        checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
        return cast(Self, Checkpoint(checkpoint_state))

    def restore(self, model: Model, optimizer: Optimizer) -> Step:
        model.load_state(self.model_state)
        optimizer.load_state(self.optimizer_state)
        return self.step

    @classmethod
    def create(
        cls,
        model: Model,
        optimizer: Optimizer,
        validation_loss: Loss,
        step: Step,
    ) -> Self:
        checkpoint_state = {
            "model": model.dump_state(),
            "optimizer": optimizer.dump_state(),
            "validation_loss": validation_loss,
            "step": step,
        }
        return cast(Self, Checkpoint(checkpoint_state))

    def save(self, checkpoint_path: ValidPath):
        checkpoint_state = {
            "model": self.model_state,
            "optimizer": self.optimizer_state,
            "validation_loss": self.validation_loss.item(),
            "step": self.step,
        }
        torch.save(checkpoint_state, checkpoint_path)


class CheckpointBuilder:
    def __init__(self, checkpoint_dir: ValidPath, checkpoint_interval: int):
        if checkpoint_interval <= 1:
            raise ValueError("Checkpoint interval should be greater than one")

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval

    def create_checkpoint(
        self, model: Model, optimizer: Optimizer, loss: Loss, step: Step
    ) -> Optional[Checkpoint]:
        return (
            Checkpoint.create(model, optimizer, loss, step)
            if self._should_checkpoint(step)
            else None
        )

    def save_checkpoint_with_step(self, checkpoint: Checkpoint):
        checkpoint_path = ValidPath.from_str(
            os.path.join(self.checkpoint_dir, f"checkpoint_{checkpoint.step}.pt")
        )
        checkpoint.save(checkpoint_path)

    def save_checkpoint_with_prefix(self, checkpoint: Checkpoint, prefix: str):
        checkpoint_path = ValidPath.from_str(
            os.path.join(self.checkpoint_dir, f"{prefix}_checkpoint.pt")
        )
        checkpoint.save(checkpoint_path)

    def _should_checkpoint(self, step: Step) -> bool:
        return int(step) % self.checkpoint_interval == 0


class CheckpointManager:
    def __init__(
        self,
        checkpoint_builder: CheckpointBuilder,
        model: Model,
        optimizer: Optimizer,
        initial_checkpoint: Optional[Checkpoint],
    ):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_builder = checkpoint_builder

        self.last_checkpoint = self._initialize_checkpoint(initial_checkpoint)
        self.best_checkpoint = self.last_checkpoint

    def update_and_save_checkpoints(self, loss: Loss, step: Step):
        self._update_checkpoints(loss, step)
        self._save_checkpoints()

    def _update_checkpoints(self, loss: Loss, step: Step):
        self.last_checkpoint = self._create_last_checkpoint(loss, step)
        self.best_checkpoint = self._define_best_checkpoint()

    def _save_checkpoints(self):
        self.checkpoint_builder.save_checkpoint_with_step(self.last_checkpoint)
        self.checkpoint_builder.save_checkpoint_with_prefix(
            self.best_checkpoint, "best"
        )

    def _initialize_checkpoint(
        self, previous_checkpoint: Optional[Checkpoint]
    ) -> Checkpoint:
        return (
            previous_checkpoint
            if previous_checkpoint
            else Checkpoint.create_initial_checkpoint(self.model, self.optimizer)
        )

    def _create_last_checkpoint(self, loss: Loss, step: Step) -> Checkpoint:
        current_checkpoint = self.checkpoint_builder.create_checkpoint(
            self.model, self.optimizer, loss, step
        )
        return current_checkpoint if current_checkpoint else self.last_checkpoint

    def _define_best_checkpoint(self) -> Checkpoint:
        if self.last_checkpoint.validation_loss < self.best_checkpoint.validation_loss:
            return self.last_checkpoint
        return self.best_checkpoint


def create_checkpoint_builder(config: TrainCheckpointConfig) -> CheckpointBuilder:
    return CheckpointBuilder(config.dir, config.interval)


# This part of the Code refers to the exception handling of the
# FullCheckpoint.load_state method, which originally was part of
# a try-except block. Nevertheless, it is not clear if this condition
# ever happens, henceforth we decided to leave it commented
# https://github.com/Edresson/SPIRA-ComParE2021/blob/369321993ef2a5358170a2593d9f1f2631886036/train.py#L147-L155
#
# class PartialCheckpoint:
#     def __init__(self, checkpoint_path: Path):
#         self.checkpoint_path = checkpoint_path
#
#     def load_state(self, model):
#         model_state = model.dump_state_dict()
#         model_state = set_init_dict(model_state, checkpoint, c)
#         model.load_state(model_state)
