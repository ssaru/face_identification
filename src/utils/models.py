from omegaconf import DictConfig

from src import models


def load_model(config: DictConfig):
    model_name = config.model.name
    model_params = config.model.params
    model = getattr(models, model_name)(**model_params)
    return model
