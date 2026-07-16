"""
KrishiMitra - Environment Verification Script

Verifies that the development environment is correctly configured.

Author: Pratiksha Malewar
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

from common.logger import LoggerManager

LoggerManager()

logger = LoggerManager.get_logger("EnvironmentVerifier")


# ==========================================================
# Package Verification
# ==========================================================

REQUIRED_PACKAGES = [
    "yaml",
    "numpy",
    "pandas",
    "cv2",
    "torch",
    "torchvision",
    "PIL",
    "matplotlib",
    "sklearn",
    "tqdm",
    "ultralytics",
]


# ==========================================================
# Project Structure
# ==========================================================

REQUIRED_DIRECTORIES = [
    "configs",
    "common",
    "datasets",
    "scripts",
    "detection",
    "segmentation",
    "classification",
    "knowledge_base",
    "pipeline",
    "saved_models",
    "outputs",
    "logs",
]

REQUIRED_CONFIGS = [
    "configs/paths.yaml",
    "configs/model.yaml",
    "configs/training.yaml",
    "configs/logging.yaml",
]


# ==========================================================
# Functions
# ==========================================================

def check_python() -> None:
    """Check Python version."""

    version = platform.python_version()

    logger.info(f"Python Version : {version}")

    if sys.version_info < (3, 11):
        logger.warning("Python 3.11 or higher is recommended.")


def check_packages() -> None:
    """Verify required Python packages."""

    logger.info("Checking required packages...")

    for package in REQUIRED_PACKAGES:

        try:
            importlib.import_module(package)
            logger.info(f"✓ {package}")

        except ImportError:
            logger.error(f"✗ {package} NOT INSTALLED")


def check_pytorch() -> None:
    """Check PyTorch installation."""

    try:
        import torch

        logger.info(f"PyTorch Version : {torch.__version__}")

        if torch.cuda.is_available():
            logger.info(f"CUDA Available : {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA not available (CPU mode).")

    except Exception as e:
        logger.error(str(e))


def check_project_structure() -> None:
    """Verify project directories."""

    logger.info("Checking project folders...")

    for directory in REQUIRED_DIRECTORIES:

        if Path(directory).exists():
            logger.info(f"✓ {directory}")

        else:
            logger.warning(f"Missing folder : {directory}")


def check_config_files() -> None:
    """Verify configuration files."""

    logger.info("Checking configuration files...")

    for config in REQUIRED_CONFIGS:

        if Path(config).exists():
            logger.info(f"✓ {config}")

        else:
            logger.error(f"Missing config : {config}")


# ==========================================================
# Main
# ==========================================================

def main():

    logger.info("=" * 60)
    logger.info("KrishiMitra Environment Verification")
    logger.info("=" * 60)

    check_python()

    check_packages()

    check_pytorch()

    check_project_structure()

    check_config_files()

    logger.info("=" * 60)
    logger.info("Environment Verification Completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()