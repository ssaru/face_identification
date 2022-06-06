from typing import Dict, List

from omegaconf import DictConfig, OmegaConf


def get_config(hparams: Dict, options: List) -> DictConfig:
    config: DictConfig = OmegaConf.create()
    for option in options:
        option_config: DictConfig = OmegaConf.load(hparams.get(option))
        config.update(option_config)
    OmegaConf.set_readonly(config, True)
    return config
