from abc import ABC, abstractmethod

import torch.nn as nn


# Loss = nn.modules.loss._Loss


class Loss(ABC):
    @abstractmethod
    def calculate(self, prediction: int, label: int) -> float:
        pass

    @abstractmethod
    def recalculate_weights(self):
        pass


class BCELoss(Loss):
    def __init__(self, reduction: str):
        super().__init__()
        self.bce_loss = nn.BCELoss(reduction=reduction)

    def calculate(self, prediction: int, label: int) -> float:
        # .item() extracts a float from a single dimension tensor
        # https://discuss.pytorch.org/t/what-is-the-difference-between-loss-and-loss-item/126083
        return self.bce_loss(prediction, label).item()

    def recalculate_weights(self):
        self.bce_loss.backward()


class ClipBCELoss(Loss):
    # TODO: Import Clip_BCE
    def __init__(self):
        super().__init__()

    def calculate(self, prediction: int, label: int) -> float:
        return 0.0

    def recalculate_weights(self):
        pass


class ClassBalancerLoss(Loss):
    pass


def define_train_loss_function(use_mixup: bool) -> Loss:
    match use_mixup:
        case True:
            return ClipBCELoss()
        case False:
            return BCELoss(reduction="none")
    raise ValueError("The use_mixup should be True or False")


def define_eval_loss_function() -> Loss:
    return BCELoss(reduction="sum")
