"""
KrishiMitra - Image Transforms (Albumentations Refactoring)

Image preprocessing and advanced augmentation for real-world generalization.
Integrates Albumentations with PyTorch dataset via a drop-in wrapper.

Author:
    Pratiksha Malewar
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ==========================================================
# Image Size & ImageNet Stats
# ==========================================================

IMAGE_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==========================================================
# Albumentations Drop-in Wrapper
# ==========================================================

class AlbumentationsWrapper:
    """
    Wraps an Albumentations transform pipeline to make it fully compatible
    with torchvision-style Dataset class calls (takes PIL Image, returns Tensor).
    """

    def __init__(self, transform: A.Compose) -> None:
        self.transform = transform

    def __call__(self, img: Image.Image) -> any:
        # Convert PIL Image to NumPy RGB array
        img_np = np.array(img)
        
        # Apply Albumentations transform
        augmented = self.transform(image=img_np)
        return augmented["image"]


# ==========================================================
# Build Pipelines
# ==========================================================

# 1. Advanced Training Augmentation Pipeline
train_aug_list = [
    A.RandomResizedCrop(size=(IMAGE_SIZE, IMAGE_SIZE), scale=(0.8, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    # ShiftScaleRotate combines RandomAffine and Rotation
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, border_mode=0, p=0.7),
    A.Perspective(scale=(0.05, 0.1), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.4),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=(3, 7)),
    ], p=0.3),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
]

# Safeguard dynamic loading of weather & dropout modules
for aug_name in ["RandomShadow", "RandomFog", "RandomRain", "Rain", "CoarseDropout"]:
    if hasattr(A, aug_name):
        aug_cls = getattr(A, aug_name)
        if aug_name == "CoarseDropout":
            train_aug_list.append(aug_cls(num_holes_range=(1, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.3))
        else:
            train_aug_list.append(aug_cls(p=0.15))

# Normalize and convert to PyTorch Tensor
train_aug_list.extend([
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
])

# 2. Validation & Test Pipeline
val_aug_list = [
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2()
]


# ==========================================================
# Expose wrapped transforms
# ==========================================================

train_transform = AlbumentationsWrapper(A.Compose(train_aug_list))
val_transform = AlbumentationsWrapper(A.Compose(val_aug_list))
test_transform = AlbumentationsWrapper(A.Compose(val_aug_list))


def get_train_transforms() -> AlbumentationsWrapper:
    """Return the Albumentations training transform wrapper."""
    return train_transform


def get_validation_transforms() -> AlbumentationsWrapper:
    """Return the Albumentations validation transform wrapper."""
    return val_transform


def get_test_transforms() -> AlbumentationsWrapper:
    """Return the Albumentations test transform wrapper."""
    return test_transform