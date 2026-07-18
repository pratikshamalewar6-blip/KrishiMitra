"""
KrishiMitra - Dataset Preprocessor

Preprocesses image classification datasets before training.

Features
--------
- Resize images
- Convert to RGB
- Preserve folder structure
- Remove corrupted images
- Generate preprocessing reports

Author:
    Pratiksha Malewar

Project:
    KrishiMitra
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List

import json
import csv

from PIL import Image, ImageOps
from tqdm import tqdm

from common.config import ConfigManager
from common.logger import LoggerManager
from common.file_utils import FileUtils


# ==========================================================
# Dataclass
# ==========================================================

@dataclass
class ImageProcessResult:
    """
    Stores preprocessing statistics
    for one dataset.
    """

    dataset: str

    total_images: int

    processed_images: int

    skipped_images: int

    failed_images: int


# ==========================================================
# Dataset Preprocessor
# ==========================================================

class DatasetPreprocessor:
    """
    Preprocess image datasets before training.
    """

    def __init__(self) -> None:

        self.config = ConfigManager()

        self.logger = LoggerManager.get_logger(
            "DatasetPreprocessor"
        )

        self.datasets = self.config.get(
            "paths.datasets"
        )

        self.output_root = (
            Path(
                self.config.get(
                    "paths.datasets.processed"
                )
            )
        )

        FileUtils.ensure_directory(
            self.output_root
        )

        # --------------------------------------
        # Image configuration
        # --------------------------------------

        self.image_size = (
            224,
            224,
        )

        self.image_format = "JPEG"

        self.image_quality = 95

        self.supported_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }

        self.logger.info(
            "Dataset Preprocessor Initialized"
        )

        # ------------------------------------------------------

    def get_dataset_paths(
        self,
    ) -> Dict[str, Path]:
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

        # ------------------------------------------------------

    def get_class_directories(
        self,
        dataset_path: Path,
    ) -> List[Path]:
        """
        Return all class folders.
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

    def create_output_directory(
        self,
        dataset_name: str,
        class_name: str,
    ) -> Path:
        """
        Create processed class directory.
        """

        output_dir = (
            self.output_root
            / dataset_name
            / class_name
        )

        FileUtils.ensure_directory(
            output_dir
        )

        return output_dir

        # ------------------------------------------------------

    def process_image(
        self,
        image_path: Path,
        output_path: Path,
    ) -> bool:
        """
        Process a single image.

        Steps
        -----
        1. Open image
        2. Convert to RGB
        3. Resize
        4. Save processed image

        Returns
        -------
        bool
            True if successful.
        """

        try:

            with Image.open(image_path) as image:

                # --------------------------
                # Convert to RGB
                # --------------------------

                if image.mode != "RGB":

                    image = image.convert("RGB")

                # --------------------------
                # Resize
                # --------------------------

                image = ImageOps.fit(
                    image,
                    self.image_size,
                    Image.Resampling.LANCZOS,
                )

                # --------------------------
                # Create output directory
                # --------------------------

                output_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                # --------------------------
                # Save image
                # --------------------------

                image.save(
                    output_path,
                    format=self.image_format,
                    quality=self.image_quality,
                )

            return True

        except Exception as error:

            self.logger.warning(
                f"Failed : {image_path.name} ({error})"
            )

            return False


        # ------------------------------------------------------

    def process_class(
        self,
        dataset_name: str,
        class_dir: Path,
    ) -> ImageProcessResult:
        """
        Process one class folder.
        """

        output_dir = self.create_output_directory(
            dataset_name,
            class_dir.name,
        )

        image_paths = sorted(
            FileUtils.list_images(class_dir)
        )

        total_images = len(image_paths)

        processed = 0
        skipped = 0
        failed = 0

        for image_path in tqdm(
            image_paths,
            desc=class_dir.name,
            leave=False,
        ):

            # output_file = (
            #     output_dir /
            #     f"{image_path.stem}.jpg"
            # )

            output_file = (
                output_dir /
                image_path.with_suffix(".jpg").name
            )

            success = self.process_image(
                image_path,
                output_file,
            )

            if success:
                processed += 1
            else:
                failed += 1

        return ImageProcessResult(
            dataset=class_dir.name,
            total_images=total_images,
            processed_images=processed,
            skipped_images=skipped,
            failed_images=failed,
        )


        # ------------------------------------------------------

    def process_dataset(
        self,
        dataset_name: str,
        dataset_path: Path,
    ) -> ImageProcessResult:
        """
        Process an entire dataset.
        """

        self.logger.info("")
        self.logger.info("=" * 70)
        self.logger.info(f"Processing Dataset : {dataset_name}")
        self.logger.info("=" * 70)

        class_dirs = self.get_class_directories(
            dataset_path
        )

        total_images = 0
        processed_images = 0
        skipped_images = 0
        failed_images = 0

        for class_dir in class_dirs:

            result = self.process_class(
                dataset_name,
                class_dir,
            )

            total_images += result.total_images
            processed_images += result.processed_images
            skipped_images += result.skipped_images
            failed_images += result.failed_images

            self.logger.info(
                f"{class_dir.name:<40}"
                f"{result.processed_images:>6} images"
            )

        return ImageProcessResult(
            dataset=dataset_name,
            total_images=total_images,
            processed_images=processed_images,
            skipped_images=skipped_images,
            failed_images=failed_images,
        )


        # ------------------------------------------------------

    def save_csv(
        self,
        result: ImageProcessResult,
    ) -> None:
        """
        Save preprocessing summary to CSV.
        """

        output_file = (
            self.output_root /
            f"{result.dataset}_preprocessing.csv"
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    asdict(result).keys()
                ),
            )

            writer.writeheader()
            writer.writerow(
                asdict(result)
            )

        self.logger.info(
            f"CSV Saved : {output_file}"
        )

        # ------------------------------------------------------

    def save_json(
        self,
        result: ImageProcessResult,
    ) -> None:
        """
        Save preprocessing summary to JSON.
        """

        output_file = (
            self.output_root /
            f"{result.dataset}_preprocessing.json"
        )

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                asdict(result),
                file,
                indent=4,
                ensure_ascii=False,
            )

        self.logger.info(
            f"JSON Saved : {output_file}"
        )

        # ------------------------------------------------------

    def print_summary(
        self,
        result: ImageProcessResult,
    ) -> None:
        """
        Print preprocessing summary.
        """

        self.logger.info("-" * 70)

        self.logger.info(
            f"Dataset           : {result.dataset}"
        )

        self.logger.info(
            f"Total Images      : {result.total_images}"
        )

        self.logger.info(
            f"Processed Images  : {result.processed_images}"
        )

        self.logger.info(
            f"Skipped Images    : {result.skipped_images}"
        )

        self.logger.info(
            f"Failed Images     : {result.failed_images}"
        )

        self.logger.info("-" * 70)


        # ------------------------------------------------------

    def run(self) -> None:
        """
        Execute dataset preprocessing.
        """

        self.logger.info("=" * 70)
        self.logger.info(
            "KrishiMitra Dataset Preprocessor"
        )
        self.logger.info("=" * 70)

        dataset_paths = self.get_dataset_paths()

        if not dataset_paths:

            self.logger.warning(
                "No datasets found."
            )

            return

        for dataset_name, dataset_path in dataset_paths.items():

            if not dataset_path.exists():

                self.logger.warning(
                    f"Dataset not found : {dataset_path}"
                )

                continue

            result = self.process_dataset(
                dataset_name,
                dataset_path,
            )

            self.print_summary(result)

            self.save_csv(result)

            self.save_json(result)

        self.logger.info("=" * 70)
        self.logger.info(
            "Dataset Preprocessing Completed Successfully"
        )
        self.logger.info("=" * 70)

# ==========================================================
# Main
# ==========================================================

def main():

    DatasetPreprocessor().run()


if __name__ == "__main__":

    main()