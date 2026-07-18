"""
KrishiMitra - PyTorch DataLoader

Creates DataLoaders for training,
validation and testing.

Author:
    Pratiksha Malewar

Project:
    KrishiMitra
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader

from common.config import ConfigManager
from common.logger import LoggerManager

from data.dataset import KrishiMitraDataset
from data.transforms import (
    get_train_transforms,
    get_validation_transforms,
)

class DataLoaderFactory:
    """
    Creates train, validation and
    test DataLoaders.
    """

    def __init__(self) -> None:

        self.logger = LoggerManager.get_logger(
            "DataLoaderFactory"
        )

        self.config = ConfigManager()

        self.batch_size = (
            self.config.get(
                "training.batch_size"
            )
            or 32
        )

        self.num_workers = (
            self.config.get(
                "training.num_workers"
            )
            or 4
        )

        self.pin_memory = (
            self.config.get(
                "training.pin_memory"
            )
            or True
        )

        self.logger.info(
            "DataLoaderFactory Initialized"
        )


    def get_split_csv(
        self,
        dataset_name: str,
        split: str,
    ) -> Path:
        """
        Return CSV path.
        """

        csv_path = (
            Path("outputs")
            / "splits"
            / f"{dataset_name}_{split}.csv"
        )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found."
            )

        return csv_path


    def create_dataset(
        self,
        dataset_name: str,
        split: str,
    ) -> KrishiMitraDataset:
        """
        Create dataset.
        """

        csv_file = self.get_split_csv(
            dataset_name,
            split,
        )

        if split == "train":

            transform = get_train_transforms()

        else:

            transform = get_validation_transforms()

        dataset = KrishiMitraDataset(
            csv_file=csv_file,
            transform=transform,
        )

        return dataset

    def create_dataloader(
        self,
        dataset_name: str,
        split: str,
    ) -> DataLoader:
        """
        Create DataLoader for one split.
        """

        dataset = self.create_dataset(
            dataset_name,
            split,
        )

        shuffle = split == "train"

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            persistent_workers=(
                self.num_workers > 0
            ),
        )

        self.logger.info(
            f"{split.capitalize()} DataLoader "
            f"created ({len(dataset)} samples)"
        )

        return loader


    def create_dataloaders(
        self,
        dataset_name: str,
    ) -> Tuple[
        DataLoader,
        DataLoader,
        DataLoader,
    ]:
        """
        Create train, validation and
        test DataLoaders.
        """

        train_loader = self.create_dataloader(
            dataset_name,
            "train",
        )

        val_loader = self.create_dataloader(
            dataset_name,
            "val",
        )

        test_loader = self.create_dataloader(
            dataset_name,
            "test",
        )

        return (
            train_loader,
            val_loader,
            test_loader,
        )

    # ==========================================================
# Main
# ==========================================================

def main():

    factory = DataLoaderFactory()

    (
        train_loader,
        val_loader,
        test_loader,
    ) = factory.create_dataloaders(
        "plantvillage"
    )

    print("=" * 60)
    print("DataLoader Test")
    print("=" * 60)

    print(f"Train Batches : {len(train_loader)}")
    print(f"Validation Batches : {len(val_loader)}")
    print(f"Test Batches : {len(test_loader)}")

    images, labels = next(iter(train_loader))

    print()
    print(f"Image Batch Shape : {images.shape}")
    print(f"Label Batch Shape : {labels.shape}")


if __name__ == "__main__":

    main()