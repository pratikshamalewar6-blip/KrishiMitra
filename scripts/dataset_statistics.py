"""
KrishiMitra - Dataset Statistics

Scans datasets and generates comprehensive statistics.

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
from PIL import Image
from tqdm import tqdm

from common.config import ConfigManager
from common.logger import LoggerManager
from common.file_utils import FileUtils
# from common.image_utils import ImageUtils


# ==========================================================
# Dataclass
# ==========================================================

@dataclass
class ClassStatistics:
    """
    Statistics for a single class.
    """

    class_name: str

    image_count: int

    average_width: float

    average_height: float

    min_width: int

    max_width: int

    min_height: int

    max_height: int


# ==========================================================
# Dataset Statistics
# ==========================================================

class DatasetStatistics:
    """
    Generate statistics for image datasets.
    """

    def __init__(self) -> None:

        # self.config = ConfigManager("configs/paths.yaml")
        self.config = ConfigManager()

        # print("=" * 60)
        # print(self.config.get("paths.outputs"))
        # print(self.config.get("paths.datasets"))
        # print("=" * 60)

        self.logger = LoggerManager.get_logger(
            "DatasetStatistics"
        )

        self.datasets = self.config.get("paths.datasets")

        self.output_dir = Path(
            self.config.get("paths.outputs")
        ) / "dataset_statistics"

        FileUtils.ensure_directory(
            self.output_dir
        )

    # ------------------------------------------------------

    # def get_dataset_paths(self) -> Dict[str, Path]:
    #     """
    #     Return configured dataset paths.
    #     """

    #     dataset_paths = {}

    #     for name, path in self.datasets.items():

    #         if name == "raw":
    #             continue

    #         dataset_paths[name] = Path(path)

    #     return dataset_paths

    def get_dataset_paths(self) -> Dict[str, Path]:

        allowed_datasets = {
            "plantvillage","plantdoc_classification","plantdoc_detection",
            }
        
        """
            Return configured dataset paths.
        """

        dataset_paths = {}

        for name, path in self.datasets.items():

            if name not in allowed_datasets:
                continue

            # if name == "raw":
            #     continue

            dataset_path = Path(path)

            # --------------------------------------------------
            # PlantVillage Classification
            # Use only color images
            # --------------------------------------------------
            if (
                name == "plantvillage"
                and (dataset_path / "color").exists()
            ):
                dataset_path = dataset_path / "color"
            # PlantDoc Classification
            # Use only training images
            if (
                name == "plantdoc_classification"
                and (dataset_path / "train").exists()
            ):
                dataset_path = dataset_path / "train"
            
            elif (
                name == "plantdoc_detection"
                and (dataset_path / "images").exists()
            ):
                dataset_path = dataset_path / "images" / "train"


            dataset_paths[name] = dataset_path

        return dataset_paths

    # ------------------------------------------------------

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

    def print_header(self):

        self.logger.info("=" * 70)

        self.logger.info(
            "KrishiMitra Dataset Statistics"
        )

        self.logger.info("=" * 70)

    # ------------------------------------------------------

    def print_footer(self):

        self.logger.info("=" * 70)

        self.logger.info(
            "Dataset Statistics Completed"
        )

        self.logger.info("=" * 70)

# ------------------------------------------------------

    def analyze_class(
        self,
        class_dir: Path,
    ) -> ClassStatistics:
        """
        Analyze a single class directory.

        Parameters
        ----------
        class_dir : Path

        Returns
        -------
        ClassStatistics
        """

        image_paths = FileUtils.list_images(class_dir)

        widths: List[int] = []
        heights: List[int] = []

        # for image_path in image_paths:

        #     metadata = ImageUtils.get_metadata(image_path)

        #     if not metadata:
        #         continue

        #     widths.append(metadata["width"])
        #     heights.append(metadata["height"])

        

        for image_path in tqdm(
            image_paths,
            desc=class_dir.name,
            leave=False,
        ):
            try:
                with Image.open(image_path) as image:
                    widths.append(image.width)
                    
                    heights.append(image.height)

            except Exception as e:
                
                # self.logger.warning(
                #     f"Skipped {image_path.name}: {e}"
                # )
                self.logger.debug(
                    f"Skipped {image_path.name}: {e}"
                    )
                
#                 self.logger.warning(
#     f"Corrupted image skipped."
# )

                continue

            # except Exception:
            #     continue

        if len(widths) == 0:

            return ClassStatistics(
                class_name=class_dir.name,
                image_count=0,
                average_width=0,
                average_height=0,
                min_width=0,
                max_width=0,
                min_height=0,
                max_height=0,
            )

        return ClassStatistics(
            class_name=class_dir.name,
            image_count=len(widths),
            average_width=round(sum(widths) / len(widths), 2),
            average_height=round(sum(heights) / len(heights), 2),
            min_width=min(widths),
            max_width=max(widths),
            min_height=min(heights),
            max_height=max(heights),
        )

# ------------------------------------------------------

    def analyze_dataset(
        self,
        dataset_name: str,
        dataset_path: Path,
    ) -> List[ClassStatistics]:
        """
        Analyze an entire dataset.

        Parameters
        ----------
        dataset_name : str
        dataset_path : Path

        Returns
        -------
        List[ClassStatistics]
        """

        self.logger.info("")
        self.logger.info(f"Dataset : {dataset_name}")
        self.logger.info(f"Location: {dataset_path}")

        if not dataset_path.exists():

            self.logger.warning(
                "Dataset does not exist."
            )

            return []

        class_dirs = self.get_class_directories(
            dataset_path
        )

        self.logger.info(
            f"Classes Found : {len(class_dirs)}"
        )

        statistics = []

        # for class_dir in class_dirs:
        
        # for class_dir in tqdm(
        #     class_dirs,desc=f"Analyzing {dataset_name}"
        # ):
        for class_dir in tqdm(
            class_dirs,
            desc=f"{dataset_name}",unit="class",
            colour="green",
            ):

            class_stats = self.analyze_class(
                class_dir
            )

            statistics.append(class_stats)

            self.logger.info(
                f"{class_stats.class_name:<35}"
                f"{class_stats.image_count:>8} images"
            )

        return statistics

# ------------------------------------------------------

    def compute_dataset_summary(
        self,
        statistics: List[ClassStatistics],
    ) -> Dict:
        """
        Compute overall dataset summary.
        """

        if len(statistics) == 0:

            return {}

        total_classes = len(statistics)

        total_images = sum(
            cls.image_count
            for cls in statistics
        )

        avg_width = round(
            sum(
                cls.average_width
                for cls in statistics
            )
            / total_classes,
            2,
        )

        avg_height = round(
            sum(
                cls.average_height
                for cls in statistics
            )
            / total_classes,
            2,
        )

        return {
            "classes": total_classes,
            "images": total_images,
            "average_width": avg_width,
            "average_height": avg_height,
        }

# ------------------------------------------------------

    def print_summary(
        self,
        summary: Dict,
    ) -> None:
        """
        Print dataset summary.
        """

        self.logger.info("-" * 70)

        self.logger.info(
            f"Classes          : {summary['classes']}"
        )

        self.logger.info(
            f"Images           : {summary['images']}"
        )

        self.logger.info(
            f"Average Width    : {summary['average_width']}"
        )

        self.logger.info(
            f"Average Height   : {summary['average_height']}"
        )

        self.logger.info("-" * 70)

# ------------------------------------------------------

    def save_csv(
        self,
        dataset_name: str,
        statistics: List[ClassStatistics],
    ) -> None:
        """
        Save class statistics to CSV.
        """

        output_file = (
            self.output_dir /
            f"{dataset_name}_statistics.csv"
        )

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8",
        ) as csvfile:
            if not statistics:
                return

            writer = csv.DictWriter(
                csvfile,
                fieldnames=list(asdict(statistics[0]).keys()),
            )

            writer.writeheader()

            for stat in statistics:
                writer.writerow(asdict(stat))

        self.logger.info(f"CSV Saved : {output_file}")

# ------------------------------------------------------

    def save_json(
        self,
        dataset_name: str,
        statistics: List[ClassStatistics],
        summary: Dict,
    ) -> None:
        """
        Save statistics as JSON.
        """

        output_file = (
            self.output_dir /
            f"{dataset_name}_statistics.json"
        )

        data = {
            "dataset": dataset_name,
            "summary": summary,
            "classes": [
                asdict(stat)
                for stat in statistics
            ],
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

        self.logger.info(f"JSON Saved : {output_file}")

# ------------------------------------------------------

    def run(self) -> None:
        """
        Execute dataset statistics generation.
        """

        self.print_header()

        dataset_paths = self.get_dataset_paths()

        for dataset_name, dataset_path in dataset_paths.items():

            statistics = self.analyze_dataset(
                dataset_name,
                dataset_path,
            )

            # if len(statistics) == 0:
            if not statistics:
                continue

            summary = self.compute_dataset_summary(
                statistics
            )

            self.print_summary(summary)

            self.save_csv(
                dataset_name,
                statistics,
            )

            self.save_json(
                dataset_name,
                statistics,
                summary,
            )

        self.print_footer()


# ==========================================================
# Main
# ==========================================================

def main():

    DatasetStatistics().run()


if __name__ == "__main__":

    main()