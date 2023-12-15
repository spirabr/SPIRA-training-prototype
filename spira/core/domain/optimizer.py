import torch

from spira.core.domain.enum import OptimizerCategory

Optimizer = torch.optim.Adam | torch.optim.AdamW | torch.optim.RAdam


def choose_optimizer(optimizer_category: OptimizerCategory) -> type[Optimizer]:
    match optimizer_category:
        case OptimizerCategory.ADAM:
            return torch.optim.Adam
        case OptimizerCategory.ADAMW:
            return torch.optim.AdamW
        case OptimizerCategory.RADAM:
            return torch.optim.RAdam
    raise ValueError("The optimizer should be Adam, AdamW or RAdam")


def build_optimizer(
    optimizer_category: OptimizerCategory, model_parameters, learning_rate, weight_decay
):
    optimizer_constructor = choose_optimizer(optimizer_category)
    return optimizer_constructor(
        model_parameters, lr=learning_rate, weight_decay=weight_decay
    )
