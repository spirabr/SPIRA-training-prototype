import random
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from typing_extensions import Self


class Random(ABC):
    def __init__(self, seed: int):
        self.seed = seed

    @abstractmethod
    def create_random(self, seed) -> Self:
        pass

    def apply_random_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        np.random.seed(self.seed)

    def get_randint_in_interval(self, first: int, second: int) -> int:
        return random.randint(first, second)

    def get_random_float_in_interval(self, first: float, second: float) -> float:
        return random.uniform(first, second)

    def choose_n_elements(self, elements: list[Any], num_elements: int) -> list[Any]:
        # TODO: deveria ser com replacement (sample faz sem replacement)
        return random.sample(elements, num_elements)


class TrainRandom(Random):
    def create_random(self, seed) -> Random:
        return self


class TestRandom(Random):
    def create_random(self, seed) -> Random:
        new_seed = self.seed * seed
        return Random(seed=new_seed)
