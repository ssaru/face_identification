import json
import logging
import logging.config
from functools import lru_cache
from pathlib import Path
from typing import Union


def set_logger(config_path: Union[str, Path] = Path("./conf/logging.json")):
    config_path = Path(config_path) if isinstance(config_path, str) else config_path
    with config_path.open("rt", encoding="utf-8") as f:
        config = json.load(f)
    logging.config.dictConfig(config)


@lru_cache
def get_logger():
    set_logger()
    logger = logging.getLogger()
    return logger
