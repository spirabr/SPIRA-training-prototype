from abc import abstractmethod
from typing import cast

import torch

from spira.adapter.config import TrainSchedulerConfig
from spira.core.domain.optimizer import Optimizer


class Scheduler(torch.optim.lr_scheduler.LRScheduler):
    @abstractmethod
    def get_lr(self):
        pass


class NoamLRScheduler(Scheduler):
    def __init__(self, optimizer: Optimizer, warmup_steps: float):
        super().__init__(optimizer.torch_optimizer)

        self.warmup_steps = float(warmup_steps)

        self.scheduler = NoamLRScheduler(optimizer, warmup_steps=warmup_steps)

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [
            base_lr
            * self.warmup_steps**0.5
            * min(step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.base_lrs
        ]


class EmptyScheduler(Scheduler):
    def __init__(self):
        super().__init__(cast(torch.optim.Optimizer, None))

    def get_lr(self):
        return

    def step(self, epoch=None):
        return


def create_scheduler(config: TrainSchedulerConfig, optimizer: Optimizer) -> Scheduler:
    if config.use_lr_decay:
        return NoamLRScheduler(optimizer, warmup_steps=config.warmup_steps)

    return EmptyScheduler()
