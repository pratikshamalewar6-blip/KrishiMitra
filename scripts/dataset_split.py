"""
KrishiMitra - Dataset Split Generator

Creates reproducible train, validation and test
splits for image classification datasets.

Author:
    Pratiksha Malewar

Project:
    KrishiMitra
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import random
from dataclasses import dataclass, asdict
import csv

from collections import defaultdict

from common.config import ConfigManager
from common.logger import LoggerManager
from common.file_utils import FileUtils
from common.seed import set_seed

# ==========================================================
# Dataclass
# ==========================================================

@dataclass
class DatasetRecord:
    """
    Represents one image record.
    """

    image_path: str

    class_name: str

    dataset: str

# ==========================================================
# Dataset Split Generator
# ==========================================================

class DatasetSplitGenerator:
    """
    Creates train/validation/test splits.
    """

    def __init__(self) -> None:

        self.config = ConfigManager()

        self.logger = LoggerManager.get_logger(
            "DatasetSplitGenerator"
        )

        set_seed(42)
        # SeedManager.set_seed()

        self.datasets = self.config.get(
            "paths.datasets"
        )

        self.output_dir = (
            Path(
                self.config.get(
                    "paths.outputs"
                )
            )
            / "splits"
        )

        FileUtils.ensure_directory(
            self.output_dir
        )


    def get_dataset_paths(self) -> Dict[str, Path]:
        """
        Return supported datasets.
        """

        allowed_datasets = {
            "plantvillage",
            "plantdoc_classification",
        }

        dataset_paths = {}

        for name, path in self.datasets.items():

            if name not in allowed_datasets:
                continue

            dataset_path = Path(path)

            if (
                name == "plantvillage"
                and (dataset_path / "color").exists()
            ):
                dataset_path = dataset_path / "color"

            if (
                name == "plantdoc_classification"
                and (dataset_path / "train").exists()
            ):
                dataset_path = dataset_path / "train"

            dataset_paths[name] = dataset_path

        return dataset_paths


    def get_class_directories(
        self,
        dataset_path: Path,
    ) -> List[Path]:
        """
        Return all class directories.
        """

        if not dataset_path.exists():
            return []

        return sorted(
            [
                directory
                for directory in dataset_path.iterdir()
                if directory.is_dir()
            ]
        )
    # ------------------------------------------------------
    """
    Collect image paths and labels.

    Returns
    -------
    List[Tuple[str, str]]
        (image_path, class_name)
    """

    # def collect_dataset_records(
    #     self,
    #     dataset_name: str,
    #     dataset_path: Path,
    # ) -> list[dict]:
    #     """
    #     Collect all image records from a dataset.

    #     Each record contains:
    #     image_path
    #     label
    #     dataset

    #     Parameters
    #     ----------
    #     dataset_name : str

    #     dataset_path : Path

    #     Returns
    #     -------
    #     list[dict]
    #     """

    #     records = []

    #     class_dirs = self.get_class_directories(
    #         dataset_path
    #     )

    #     for class_dir in class_dirs:

    #         label = class_dir.name

    #         image_paths = FileUtils.list_images(
    #             class_dir
    #         )

    #         for image_path in image_paths:

    #             records.append(
    #                 {
    #                     "image_path": str(image_path),
    #                     "label": label,
    #                     "dataset": dataset_name,
    #                 }
    #             )

    #     self.logger.info(
    #         f"{dataset_name}: Collected "
    #         f"{len(records)} images."
    #     )

    #     return records
    

    
    def collect_records(
        self,
        dataset_name: str,
        dataset_path: Path,
    ) -> List[DatasetRecord]:
        """
        Collect all image records.
        """

        records = []

        class_dirs = self.get_class_directories(
            dataset_path
        )

        for class_dir in class_dirs:

            image_paths = sorted(FileUtils.list_images(
                class_dir
            ))

            for image_path in image_paths:

                records.append(
                    DatasetRecord(
                        image_path=str(image_path),
                        class_name=class_dir.name,
                        dataset=dataset_name,
                    )
                )

        self.logger.info(
            f"Collected {len(records)} images."
        )

        return records
    
        # ------------------------------------------------------

    def split_records(
        self,
        records: List[DatasetRecord],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, List[DatasetRecord]]:
        """
        Split dataset into train, validation and test.
        """

        # Group records by class

        # class_records = {}

        # for record in records:

        #     class_name = record.class_name

        #     if class_name not in class_records:

        #         class_records[class_name] = []

        #     class_records[class_name].append(record)
        
        if not (
            train_ratio + val_ratio + test_ratio == 1.0
            ):
            raise ValueError(
                "Train, validation and test ratios must sum to 1.0"
            )
        
        if not records:
            return {
                "train": [],
                "val": [],
                "test": [],
            }
   

        # ----------------------------------------
        # Group records by class
        # ----------------------------------------

        class_records = defaultdict(list)

        for record in records:
            class_records[record.class_name].append(record)

        train_records = []
        val_records = []
        test_records = []



        # ----------------------------------------
        # Split each class separately
        # ----------------------------------------


        for class_name, class_list in class_records.items():

            class_list = class_list.copy()

            random.shuffle(class_list)

            total = len(class_list)

            train_end = int(total * train_ratio)
            val_end = train_end + int(total * val_ratio)

            train_records.extend(
                class_list[:train_end]
            )

            val_records.extend(
                class_list[train_end:val_end]
            )

            test_records.extend(
                class_list[val_end:]
            )

        # ----------------------------------------
        # Shuffle final splits
        # ----------------------------------------

        random.shuffle(train_records)
        random.shuffle(val_records)
        random.shuffle(test_records)

        return {
            "train": train_records,
            "val": val_records,
            "test": test_records,
        }
    

        # ------------------------------------------------------

    
    def save_split(
        self,
        dataset_name: str,
        split_name: str,
        records: List[DatasetRecord],
    ) -> None:
        
        if not records:
            self.logger.warning(
                f"No records found for {split_name} split."
            )
            return

        """
        Save split CSV.
        """

        output_file = (
            self.output_dir /
            f"{dataset_name}_{split_name}.csv"
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:

            writer = csv.DictWriter(
                csvfile,
                fieldnames=list(
                    asdict(records[0]).keys()
                ),
            )

            writer.writeheader()

            for record in records:
                writer.writerow(asdict(record))

        self.logger.info(
            f"{split_name.upper()} CSV Saved : {output_file}"
        )


        # ------------------------------------------------------

    def print_split_summary(
        self,
        splits: Dict[str, List[DatasetRecord]],
    ) -> None:
        """
        Print split summary.
        """

        self.logger.info("-" * 60)

        for split_name, records in splits.items():

            self.logger.info(
                f"{split_name:<10}: {len(records)} images"
            )

        self.logger.info("-" * 60)

        # ------------------------------------------------------

    def process_dataset(
        self,
        dataset_name: str,
        dataset_path: Path,
    ) -> None:
        """
        Process one dataset.
        """

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"Dataset : {dataset_name}")
        self.logger.info("=" * 70)

        records = self.collect_records(
            dataset_name,
            dataset_path,
        )

        splits = self.split_records(records)

        self.print_split_summary(splits)

        for split_name, split_records in splits.items():

            if not split_records:
                continue

            self.save_split(
                dataset_name,
                split_name,
                split_records,
            )

        # ------------------------------------------------------

    def run(self) -> None:
        """
        Execute dataset split generation.
        """

        dataset_paths = self.get_dataset_paths()

        for dataset_name, dataset_path in dataset_paths.items():

            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info(dataset_name)
            self.logger.info("=" * 60)


            records = self.collect_records(
                dataset_name,
                dataset_path,
            )

            splits = self.split_records(records)

            train_records = splits["train"]
            val_records = splits["val"]
            test_records = splits["test"]

            self.logger.info(
                f"Total Records : {len(records)}"
            )

            self.logger.info(
                f"Train : {len(train_records)}"
            )

            self.logger.info(
                f"Validation : {len(val_records)}"
            )

            self.logger.info(
                f"Test : {len(test_records)}"
            )

            self.save_split(
                dataset_name,
                "train",
                train_records,
            )

            self.save_split(
                dataset_name,
                "val",
                val_records,
            )

            self.save_split(
                dataset_name,
                "test",
                test_records,
            )
            # self.logger.info(
            #     f"Total Records : {len(records)}"
            # )

            # self.logger.info(
            #     f"Train : {len(splits['train'])}"
            # )

            # self.logger.info(
            #     f"Validation : {len(splits['val'])}"
            # )

            # self.logger.info(
            #     f"Test : {len(splits['test'])}"
            # )

            # self.save_split(
            #     dataset_name,
            #     "train",
            #     splits["train"],
            # )

            # self.save_split(
            #     dataset_name,
            #     "val",
            #     splits["val"],
            # )

            # self.save_split(
            #     dataset_name,
            #     "test",
            #     splits["test"],
            # )

    # ==========================================================



    
# Main
# ==========================================================

def main():

    DatasetSplitGenerator().run()


if __name__ == "__main__":

    main()