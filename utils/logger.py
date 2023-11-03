import logging.config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

BASE_LOGGER_CONFIG = {
    "version": 1,
    "disable_exisiting_loggers": False,
    "formatters": {
        "simple": {"format": r"%(message)s"},
        "datetime": {"format": r"%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",                   # print every thing on screen
            "level": "DEBUG",
            "formatter": "simple",
            "stream": "ext://sys.stdout"
        },
        "info_file_handler": {
            "class": "logging.handlers.RotatingFileHandler",     # log info level and above into file (i.e. ignore DEBUG)
            "level": "INFO",
            "formatter": "datetime",
            "filename": "info.log",
            "maxBytes": 10485760,
            "backupCount": 20,
            "encoding": "utf8"
        }
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "info_file_handler"]
    }
}


def setup_logging(
    log_root: "Path",
) -> None:
    """ set up logging module into both console on screen and into log file"""
    config = BASE_LOGGER_CONFIG
    for handler_config in config["handlers"].values():
        if "filename" in handler_config:
            handler_config["filename"] = str(log_root / handler_config["filename"])
    logging.config.dictConfig(config)
