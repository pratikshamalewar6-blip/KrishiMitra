"""
KrishiMitra - PyTorch Dataset

Loads processed images using CSV split files.

Author:
    Pratiksha Malewar

Project:
    KrishiMitra
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from PIL import Image

from torch.utils.data import Dataset

from common.logger import LoggerManager

# ==========================================================
# Dataclass
# ==========================================================

@dataclass
class DatasetSample:
    """
    Represents one dataset sample.
    """

    image_path: str

    class_name: str

    dataset: str


# ==========================================================
# Dataset
# ==========================================================

class KrishiMitraDataset(Dataset):
    """
    PyTorch Dataset for disease classification.
    """

    def __init__(
        self,
        csv_file: str | Path,
        transform=None,
    ) -> None:

        self.logger = LoggerManager.get_logger(
            "KrishiMitraDataset"
        )

        self.transform = transform

        if not Path(csv_file).exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_file}"
            )

        self.dataframe = pd.read_csv(
            csv_file
        )

        self.samples = [
            DatasetSample(
                image_path=row.image_path,
                class_name=row.class_name,
                dataset=row.dataset,
            )
            for row in self.dataframe.itertuples()
        ]

        self.class_to_index = (
            self.build_class_mapping()
        )

        self.index_to_class = {
            index: class_name
            for class_name, index in self.class_to_index.items()
        }

        self.logger.info(
            f"Loaded {len(self.samples)} samples."
        )


        # ------------------------------------------------------

    def __len__(
        self,
    ) -> int:
        """
        Return dataset size.
        """

        return len(self.samples)


        # ------------------------------------------------------

    def build_class_mapping(
        self,
    ) -> Dict[str, int]:
        """
        Create class-to-index mapping.
        """

        classes = sorted(
            list(
                {
                    sample.class_name
                    for sample in self.samples
                }
            )
        )

        return {
            class_name: index
            for index, class_name
            in enumerate(classes)
        }

    
        # ------------------------------------------------------

    def __getitem__(
        self,
        index: int,
    ):
        """
        Return one dataset sample.
        """

        sample = self.samples[index]

        image_path = Path(sample.image_path)

        # -------------------------------
        # Load Image
        # -------------------------------

        try:
            image = Image.open(
                image_path
            ).convert("RGB")
        except Exception as e:
            self.logger.error(
                f"Failed to load image: {image_path}"
            )
            raise RuntimeError(
                f"Failed to load image: {image_path}"
            ) from e

        # -------------------------------
        # Apply Transform
        # -------------------------------

        if self.transform is not None:

            image = self.transform(
                image
            )

        # -------------------------------
        # Convert Label
        # -------------------------------

        label = self.class_to_index[
            sample.class_name
        ]

        return image, label

    
        # ------------------------------------------------------

    def get_classes(
        self,
    ) -> List[str]:
        """
        Return class names.
        """

        return list(
            self.class_to_index.keys()
        )

    
        # ------------------------------------------------------

    @property
    def num_classes(
        self,
    ) -> int:
        """
        Return number of classes.
        """

        return len(
            self.class_to_index
        )


        # ------------------------------------------------------

    def get_class_mapping(
        self,
    ) -> Dict[str, int]:
        """
        Return class mapping.
        """

        return self.class_to_index


        # ------------------------------------------------------

    def get_index_mapping(
        self,
    ) -> Dict[int, str]:
        """
        Return index-to-class mapping.
        """

        return self.index_to_class


    # ==========================================================
# Main
# ==========================================================

def main():

    csv_file = (
        Path("outputs")
        / "splits"
        / "plantvillage_train.csv"
    )

    dataset = KrishiMitraDataset(
        csv_file=csv_file,
    )

    print("=" * 60)
    print("Dataset Test")
    print("=" * 60)

    print(f"Samples  : {len(dataset)}")
    print(f"Classes  : {dataset.num_classes}")

    image, label = dataset[0]

    print(type(image))
    print(label)


if __name__ == "__main__":

    main()