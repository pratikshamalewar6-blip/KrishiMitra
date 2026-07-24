"""
KrishiMitra - Classification Dataset Adapter

Exposes the core dataset, transforms, and data loaders for disease classification.

Author:
    Antigravity AI
"""

from __future__ import annotations

# Import and expose the core data management classes and utilities
from data.dataset import KrishiMitraDataset, DatasetSample
from data.dataloader import DataLoaderFactory
from data.transforms import (
    get_train_transforms,
    get_validation_transforms,
    get_test_transforms,
)
