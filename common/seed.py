"""
KrishiMitra - Random Seed Utility

Ensures reproducible experiments across Python, NumPy and PyTorch.

Author: Pratiksha Malewar
"""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from common.logger import LoggerManager


logger = LoggerManager.get_logger("Seed")


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducible experiments.

    Args:
        seed (int): Random seed value.
    """

    logger.info(f"Setting random seed to {seed}")

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Python Hash Seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:

        # PyTorch CPU
        torch.manual_seed(seed)

        # PyTorch GPU
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # cuDNN Settings
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        logger.info("PyTorch seed configured successfully.")

    else:
        logger.warning("PyTorch is not installed. Only Python and NumPy seeds were configured.")