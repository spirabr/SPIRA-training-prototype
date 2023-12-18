from abc import ABC, abstractmethod

import torch

from spira.adapter.config import Config
from spira.core.domain.optimizer import Optimizer


class Scheduler(ABC):
    @abstractmethod
    def step(self):
        pass


class NoamLRScheduler(torch.optim.lr_scheduler._LRScheduler, Scheduler):
    def __init__(self, optimizer, warmup_steps=0.1, last_epoch=-1):
        self.warmup_steps = float(warmup_steps)
        super(NoamLRScheduler, self).__init__(optimizer, last_epoch)
        self.scheduler = NoamLRScheduler(
            optimizer, warmup_steps=warmup_steps, last_epoch=last_epoch
        )

    def step(self):
        self.scheduler.step()

    def get_lr(self):
        step = max(self.last_epoch, 1)
        return [
            base_lr
            * self.warmup_steps**0.5
            * min(step * self.warmup_steps**-1.5, step**-0.5)
            for base_lr in self.base_lrs
        ]


class EmptyScheduler(Scheduler):
    def step(self):
        return


def create_scheduler(c: Config, optimizer: Optimizer) -> Scheduler:
    if c.train_config["lr_decay"]:
        return NoamLRScheduler(
            optimizer, warmup_steps=c.train_config["warmup_steps"], last_epoch=step - 1
        )

    return EmptyScheduler()
