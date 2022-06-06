from typing import Tuple

from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset


def get_dataloaders(
    dataloader_conf: DictConfig,
    train_dataset: Dataset,
    valid_dataset: Dataset,
) -> Tuple[DataLoader, DataLoader]:

    train_loader_params = dict(dataloader_conf.train.params)
    train_loader_params.update({"dataset": train_dataset})
    train_loader = DataLoader(**train_loader_params)

    valid_loader_params = dict(dataloader_conf.valid.params)
    valid_loader_params.update({"dataset": valid_dataset})
    valid_loader = DataLoader(**valid_loader_params)

    return train_loader, valid_loader
