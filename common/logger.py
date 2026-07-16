"""
KrishiMitra - Centralized Logging Utility

Provides consistent logging across the entire project.

Author: Pratiksha Malewar
"""

from __future__ import annotations

import logging
from pathlib import Path

from common.config import ConfigManager


class LoggerManager:
    """
    Creates and manages project-wide loggers.
    """

    _initialized = False

    def __init__(self, config_path: str = "configs/logging.yaml") -> None:

        if LoggerManager._initialized:
            return

        config = ConfigManager(config_path)

        log_dir = Path(
            config.get("logging.file.directory", "logs")
        )

        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / config.get(
            "logging.file.filename",
            "krishimitra.log"
        )

        log_level = getattr(
            logging,
            config.get("logging.level", "INFO").upper()
        )

        formatter = logging.Formatter(
            fmt=config.get("logging.formatter.format"),
            datefmt=config.get("logging.formatter.date_format"),
        )

        root_logger = logging.getLogger()

        root_logger.setLevel(log_level)

        if not root_logger.handlers:

            file_handler = logging.FileHandler(
                log_file,
                encoding="utf-8"
            )

            file_handler.setFormatter(formatter)

            console_handler = logging.StreamHandler()

            console_handler.setFormatter(formatter)

            root_logger.addHandler(file_handler)
            root_logger.addHandler(console_handler)

        LoggerManager._initialized = True

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Returns a configured logger.
        """
        return logging.getLogger(name)