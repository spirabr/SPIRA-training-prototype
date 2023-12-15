from abc import ABC, abstractmethod
from pathlib import Path

import torch


class Checkpoint(ABC):
    @abstractmethod
    def load_state(self, model):
        pass


class EmptyCheckpoint:
    def load_state(self, model):
        return


class FullCheckpoint:
    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path

    def load_state(self, model):
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])


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
#         model_state = model.state_dict()
#         model_state = set_init_dict(model_state, checkpoint, c)
#         model.load_state_dict(model_state)
