import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import yaml
from datetime import datetime


class LoggerSetup:
    _loggers = {}

    @staticmethod
    def get_logger(name: str, log_dir: str = None) -> logging.Logger:
        if name in LoggerSetup._loggers:
            return LoggerSetup._loggers[name]

        if log_dir is None:
            log_dir = "./outputs/logs"

        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_file = log_path / f"{name}-{timestamp}.log"

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10485760,
            backupCount=5
        )
        file_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

        LoggerSetup._loggers[name] = logger
        return logger
