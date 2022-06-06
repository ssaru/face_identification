"""
Usage:
    main.py train [options] [--config=<config path>] [--checkpoint_path=<checkpoint path>]
    main.py train (-h | --help)
Options:
    --config <config path>  Path to YAML file for model configuration  [default: conf/resnet50.yaml] [type: path]
    --checkpoint-path <checkpoint path>  Path to model weight for resume  [default: None] [type: path]
    -h --help  Show this.
"""

from typing import Dict

from omegaconf import DictConfig
from pytorch_lightning import Trainer

from src.core.container.trainer import TrainingContainer
from src.utils import get_config, get_dataloaders, get_dataset, get_logger, load_model

logger = get_logger()


def train(hparams: Dict):
    config_list = ["--config"]
    config: DictConfig = get_config(hparams=hparams, options=config_list)
    logger.info(f"config: {config}")

    checkpoint_path = str(hparams.get("--checkpoint-path"))
    checkpoint_path = checkpoint_path if (checkpoint_path != "None") else None
    logger.info(f"checkpoint_path: {checkpoint_path}")

    train_dataset, valid_dataset = get_dataset(dataset_conf=config.dataset)
    train_dataloader, valid_dataloader = get_dataloaders(
        dataloader_conf=config.dataloader,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    )
    model = load_model(config=config)
    logger.info(f"models: {model}")

    model_container = TrainingContainer(model=model, config=config)
    logger.info(f"model container: {model_container}")

    trainer_conf = config.trainer
    trainer_params = trainer_conf.params
    trainer = Trainer(**trainer_params)

    trainer.fit(
        model=model_container,
        train_dataloaders=train_dataloader,
        val_dataloaders=valid_dataloader,
    )
