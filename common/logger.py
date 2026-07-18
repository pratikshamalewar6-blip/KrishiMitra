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

    def __init__(self) -> None:

        # Prevent multiple initializations
        if LoggerManager._initialized:
            return

        # Load all project configurations
        config = ConfigManager()

        # ---------------------------------------------------------
        # Logging Configuration
        # ---------------------------------------------------------

        log_dir = Path(
            config.get("logging.file.directory", "logs")
        )

        log_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        log_file = log_dir / config.get(
            "logging.file.filename",
            "krishimitra.log"
        )

        log_level = getattr(
            logging,
            config.get("logging.level", "INFO").upper()
        )

        formatter = logging.Formatter(
            fmt=config.get(
                "logging.formatter.format",
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            ),
            datefmt=config.get(
                "logging.formatter.date_format",
                "%Y-%m-%d %H:%M:%S"
            ),
        )

        root_logger = logging.getLogger()

        root_logger.setLevel(log_level)

        # Avoid duplicate handlers
        if not root_logger.handlers:

            # File Handler
            if config.get("logging.file.enabled", True):

                file_handler = logging.FileHandler(
                    log_file,
                    encoding="utf-8"
                )

                file_handler.setFormatter(formatter)

                root_logger.addHandler(file_handler)

            # Console Handler
            if config.get("logging.console.enabled", True):

                console_handler = logging.StreamHandler()

                console_handler.setFormatter(formatter)

                root_logger.addHandler(console_handler)

        LoggerManager._initialized = True

    # ---------------------------------------------------------

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Returns a configured logger.
        """

        if not LoggerManager._initialized:
            LoggerManager()

        return logging.getLogger(name)