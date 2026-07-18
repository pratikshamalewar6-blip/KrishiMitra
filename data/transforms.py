"""
KrishiMitra - Image Transforms

Image preprocessing and augmentation for training
and validation datasets.

Author:
    Pratiksha Malewar

Project:
    KrishiMitra
"""

from __future__ import annotations

# pyrefly: ignore [missing-import]
from torchvision import transforms

# ==========================================================
# Image Size
# ==========================================================

IMAGE_SIZE = 224

# ==========================================================
# ImageNet Statistics
# ==========================================================

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)

# ==========================================================
# Training Transform
# ==========================================================

train_transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.RandomRotation(
            degrees=15
        ),

        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
            saturation=0.20,
            hue=0.05,
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)

# ==========================================================
# Validation Transform
# ==========================================================

val_transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)

# ==========================================================
# Test Transform
# ==========================================================

test_transform = transforms.Compose(
    [
        transforms.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)

# ==========================================================
# Transform Getter Functions
# ==========================================================

def get_train_transforms():
    """Return the Compose transform object for training dataset."""
    return train_transform


def get_validation_transforms():
    """Return the Compose transform object for validation dataset."""
    return val_transform


def get_test_transforms():
    """Return the Compose transform object for test dataset."""
    return test_transform