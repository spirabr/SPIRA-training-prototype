from enum import Enum

import torch
from torch.optim import Adam, AdamW, RAdam

from spira.adapter.config import TrainOptimizerConfig
from spira.core.domain.model import Parameter


class OptimizerCategory(Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    RADAM = "radam"


class Optimizer:
    def __init__(self, torch_optimizer: torch.optim.Optimizer):
        self.torch_optimizer = torch_optimizer

    def zero_grad(self):
        self.torch_optimizer.zero_grad()

    def step(self):
        return self.torch_optimizer.step()

    def dump_state(self) -> dict:
        return self.torch_optimizer.state_dict()

    def load_state(self, state: dict):
        self.torch_optimizer.load_state_dict(state)


def _create_torch_optimizer(
    config: TrainOptimizerConfig,
    model_parameters: list[Parameter],
) -> torch.optim.Optimizer:
    match config.category:
        case OptimizerCategory.ADAM:
            return Adam(
                model_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        case OptimizerCategory.ADAMW:
            return AdamW(
                model_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        case OptimizerCategory.RADAM:
            return RAdam(
                model_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
    raise ValueError(f"Unknown optimizer category {config.category}")


def create_optimizer(
    config: TrainOptimizerConfig,
    model_parameters: list[Parameter],
) -> Optimizer:
    return Optimizer(_create_torch_optimizer(config, model_parameters))
