import torch.nn as nn

def define_loss_function(use_mixup: bool):
    match use_mixup:
        case True:
            return Clip_BCE()
        case False:
            return nn.BCELoss()
    raise ValueError("The use_mixup should be True or False")
