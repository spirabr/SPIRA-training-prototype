from torch.utils.data import DataLoader

from spira.core.domain.dataset import Dataset


def create_train_dataloader(
    train_dataset: Dataset, batch_size: int, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
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
        shuffle=False,
    )
