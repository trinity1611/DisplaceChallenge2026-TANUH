import logging
import sys
import os
from typing import Optional


class LoggerSetup:
    """
    Centralized logger setup.
    Ensures consistent formatting and avoids duplicate handlers.
    """

    @staticmethod
    def get_logger(
        name: str,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
    ) -> logging.Logger:
        logger = logging.getLogger(name)

        if logger.handlers:
            return logger

        logger.setLevel(level)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

      
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False
        return logger
