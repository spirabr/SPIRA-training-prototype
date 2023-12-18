from typing import cast

import torch
from torch.utils.data import DataLoader

from spira.core.domain.audio import Audio, get_wavs_from_audios
from spira.core.domain.dataset import Dataset


def create_train_dataloader(
    train_dataset: Dataset, batch_size: int, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_basic_collate_fn,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        sampler=None,
    )


def create_test_dataloader(
    test_dataset: Dataset, batch_size: int, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_basic_collate_fn,
        shuffle=False,
    )


def _basic_collate_fn(
    batch: list[tuple[Audio, int]]
) -> tuple[list[torch.Tensor], list[int]]:
    audios, labels = cast(tuple[list[Audio], list[int]], zip(*batch))
    return get_wavs_from_audios(audios), labels
