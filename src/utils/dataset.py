from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.core import datasets


def get_dataset(dataset_conf: DictConfig) -> Dataset:
    train_dataset_conf = dataset_conf.train
    valid_dataset_conf = dataset_conf.valid

    train_dataset_name = train_dataset_conf.name
    train_dataset_params = train_dataset_conf.params
    valid_dataset_name = valid_dataset_conf.name
    valid_dataset_params = valid_dataset_conf.params

    train_dataset = getattr(datasets, train_dataset_name)(**train_dataset_params)
    valid_dataset = getattr(datasets, valid_dataset_name)(**valid_dataset_params)

    return train_dataset, valid_dataset
