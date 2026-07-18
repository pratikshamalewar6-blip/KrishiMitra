"""
KrishiMitra - Dataset Validator

Validates image classification datasets before training.

Checks:
- Dataset exists
- Class folders exist
- Empty classes
- Corrupted images
- Invalid image extensions
- Duplicate filenames

Author:
    Pratiksha Malewar

Project:
    KrishiMitra
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

from PIL import Image
from tqdm import tqdm

from common.config import ConfigManager
from common.file_utils import FileUtils
from common.logger import LoggerManager


# ==========================================================
# Dataclass
# ==========================================================

@dataclass
class ValidationResult:
    """
    Validation summary for one dataset.
    """

    dataset: str

    total_classes: int

    total_images: int

    corrupted_images: int

    empty_classes: int

    duplicate_images: int

    invalid_extensions: int

    passed: bool


# ==========================================================
# Dataset Validator
# ==========================================================

class DatasetValidator:

    def __init__(self) -> None:

        self.config = ConfigManager()

        self.logger = LoggerManager.get_logger(
            "DatasetValidator"
        )

        self.datasets = self.config.get(
            "paths.datasets"
        )

        self.output_dir = (
            Path(
                self.config.get(
                    "paths.outputs"
                )
            )
            / "dataset_validation"
        )

        FileUtils.ensure_directory(
            self.output_dir
        )

        self.allowed_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tif",
            ".tiff",
            ".webp",
        }

    # ------------------------------------------------------

    def get_dataset_paths(self) -> Dict[str, Path]:

        allowed = {
            "plantvillage",
            "plantdoc_classification",
        }

        dataset_paths = {}

        for name, path in self.datasets.items():

            if name not in allowed:
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

    def compute_file_hash(self, file_path: Path) -> str:
        """
        Compute MD5 hash of a file.
        """
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    # ------------------------------------------------------

    def validate_image(self, image_path: Path) -> bool:
        """
        Verify if image can be opened and is not corrupted.
        """
        try:
            with Image.open(image_path) as image:
                image.verify()
            return True
        except Exception:
            return False

    # ------------------------------------------------------

    def validate_dataset(
        self,
        dataset_name: str,
        dataset_path: Path,
    ) -> ValidationResult:

        self.logger.info("-" * 60)
        self.logger.info(f"Validating {dataset_name}")
        self.logger.info("-" * 60)

        total_classes = 0
        total_images = 0
        corrupted_images = 0
        empty_classes = 0
        duplicate_images = 0
        invalid_extensions = 0

        if not dataset_path.exists():
            self.logger.warning(
                f"{dataset_name} folder not found at: {dataset_path}"
            )
            return ValidationResult(
                dataset=dataset_name,
                total_classes=0,
                total_images=0,
                corrupted_images=0,
                empty_classes=0,
                duplicate_images=0,
                invalid_extensions=0,
                passed=False,
            )

        class_dirs = self.get_class_directories(
            dataset_path
        )

        total_classes = len(class_dirs)
        if not class_dirs:
            self.logger.warning(
                f"No class folders found in {dataset_name}."
            )
            return ValidationResult(
                dataset=dataset_name,
                total_classes=0,
                total_images=0,
                corrupted_images=0,
                empty_classes=0,
                duplicate_images=0,
                invalid_extensions=0,
                passed=False,
            )

        seen_hashes = set()

        for class_dir in class_dirs:

            all_files = [x for x in class_dir.iterdir() if x.is_file()]

            if not all_files:
                empty_classes += 1
                continue

            for file_path in tqdm(
                all_files,
                desc=class_dir.name,
                leave=False,
            ):
                total_images += 1

                # 1. Extension Check
                ext = file_path.suffix.lower()
                if ext not in self.allowed_extensions:
                    invalid_extensions += 1

                # 2. Duplicate Check
                file_hash = self.compute_file_hash(file_path)
                if file_hash in seen_hashes:
                    duplicate_images += 1
                else:
                    seen_hashes.add(file_hash)

                # 3. Corruption Check
                if ext in self.allowed_extensions:
                    if not self.validate_image(file_path):
                        corrupted_images += 1

        passed = (
            corrupted_images == 0
            and empty_classes == 0
            and duplicate_images == 0
            and invalid_extensions == 0
        )

        return ValidationResult(
            dataset=dataset_name,
            total_classes=total_classes,
            total_images=total_images,
            corrupted_images=corrupted_images,
            empty_classes=empty_classes,
            duplicate_images=duplicate_images,
            invalid_extensions=invalid_extensions,
            passed=passed,
        )

    # ------------------------------------------------------

    def save_csv(
        self,
        dataset_name: str,
        result: ValidationResult,
    ) -> None:

        output_file = (
            self.output_dir
            / f"{dataset_name}_validation_report.csv"
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=list(asdict(result).keys()),
            )

            writer.writeheader()
            writer.writerow(asdict(result))

        self.logger.info(
            f"Validation CSV Saved : {output_file}"
        )

    # ------------------------------------------------------

    def save_json(
        self,
        dataset_name: str,
        result: ValidationResult,
    ) -> None:

        output_file = (
            self.output_dir
            / f"{dataset_name}_validation_report.json"
        )

        data = {
            "dataset": dataset_name,
            "summary": asdict(result)
        }

        with open(
            output_file,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                ensure_ascii=False,
            )

        self.logger.info(
            f"Validation JSON Saved: {output_file}"
        )

    # ------------------------------------------------------

    def print_summary(self, result: ValidationResult) -> None:

        self.logger.info("-" * 60)
        self.logger.info(f"Classes          : {result.total_classes}")
        self.logger.info(f"Images           : {result.total_images}")
        self.logger.info(f"Corrupted Images : {result.corrupted_images}")
        self.logger.info(f"Empty Classes    : {result.empty_classes}")
        self.logger.info(f"Duplicate Images : {result.duplicate_images}")
        self.logger.info(f"Invalid Exts     : {result.invalid_extensions}")

        if result.passed:
            self.logger.info("Validation Status: PASSED")
        else:
            self.logger.warning("Validation Status: FAILED")
        self.logger.info("-" * 60)

    # ------------------------------------------------------

    def run(self) -> None:

        self.logger.info("=" * 70)
        self.logger.info(
            "KrishiMitra Dataset Validator"
        )
        self.logger.info("=" * 70)

        dataset_paths = self.get_dataset_paths()

        for dataset_name, dataset_path in dataset_paths.items():

            result = self.validate_dataset(
                dataset_name,
                dataset_path,
            )

            self.print_summary(result)

            self.save_csv(
                dataset_name,
                result,
            )

            self.save_json(
                dataset_name,
                result,
            )

        self.logger.info("=" * 70)
        self.logger.info(
            "Dataset Validation Completed"
        )
        self.logger.info("=" * 70)


# ==========================================================
# Main
# ==========================================================

def main() -> None:

    DatasetValidator().run()


if __name__ == "__main__":

    main()