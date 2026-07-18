"""
============================================================
KrishiMitra
Dataset Verification Script

Author : Pratiksha Malewar
============================================================
"""

from pathlib import Path
from collections import defaultdict
from PIL import Image
from tqdm import tqdm
import json

from common.logger import LoggerManager
from common.config import ConfigManager

logger = LoggerManager.get_logger("DatasetVerifier")
config = ConfigManager()

# ---------------------------------------------------
# Dataset Paths
# ---------------------------------------------------

# plantvillage_path = Path(config.get("datasets.plantvillage"))
# plantdoc_path = Path(config.get("datasets.plantdoc"))

# report_folder = Path(config.get("outputs.dataset_reports"))

# plantvillage_path = Path(
#     config.get("paths.paths.datasets.plantvillage")
# )

# plantdoc_path = Path(
#     config.get("paths.paths.datasets.plantdoc_detection")
# )

# report_folder = Path(
#     config.get("paths.paths.outputs")
# ) / "dataset_reports"
# ---------------------------------------------------
# Dataset Paths
# ---------------------------------------------------

plantvillage_path = Path(
    config.get("paths.datasets.plantvillage")
)

plantdoc_path = Path(
    config.get("paths.datasets.plantdoc_detection")
)

report_folder = (
    Path(config.get("paths.outputs"))
    / "dataset_reports"
)

report_folder.mkdir(
    parents=True,
    exist_ok=True
)

manifest_file = (
    report_folder
    / "dataset_manifest.json"
)

report_folder.mkdir(parents=True, exist_ok=True)

manifest_file = report_folder / "dataset_manifest.json"

# ---------------------------------------------------
# Statistics
# ---------------------------------------------------

statistics = {
    "PlantVillage": {},
    "PlantDoc": {}
}

corrupted_images = []

# ---------------------------------------------------
# Verification Function
# ---------------------------------------------------

def verify_dataset(dataset_path: Path, dataset_name: str):

    logger.info("=" * 60)
    logger.info(f"Checking {dataset_name}")
    logger.info("=" * 60)

    class_counts = defaultdict(int)

    total_images = 0

    if not dataset_path.exists():
        logger.error(f"{dataset_name} folder not found.")
        return

    class_folders = sorted(
        [x for x in dataset_path.iterdir() if x.is_dir()]
    )

    logger.info(f"Found {len(class_folders)} classes.")

    for class_folder in class_folders:

        # image_files = list(class_folder.glob("*"))
        image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp"
        }

        image_files = [
            file
            for file in class_folder.iterdir()
            if file.suffix.lower() in image_extensions
        ]

        class_counts[class_folder.name] = len(image_files)

        for image_path in tqdm(
                image_files,
                desc=class_folder.name,
                leave=False):

            try:
                with Image.open(image_path) as img:
                    img.verify()

                # img = Image.open(image_path)
                # img.verify()

                total_images += 1

            except Exception as e:
                
                logger.warning(
                    f"Corrupted image: {image_path}"
                    )

                logger.warning(str(e))

                corrupted_images.append(str(image_path))
            # except Exception:

            #     corrupted_images.append(str(image_path))

    statistics[dataset_name]["classes"] = len(class_folders)
    statistics[dataset_name]["images"] = total_images
    statistics[dataset_name]["class_distribution"] = dict(class_counts)

    logger.info(f"Classes : {len(class_folders)}")
    logger.info(f"Images  : {total_images}")

# ---------------------------------------------------
# Run Verification
# ---------------------------------------------------

logger.info("Starting Dataset Verification")

verify_dataset(plantvillage_path, "PlantVillage")
verify_dataset(plantdoc_path, "PlantDoc")

statistics["Corrupted Images"] = corrupted_images

with open(
    manifest_file,
    "w",
    encoding="utf-8"
) as f:
# with open(manifest_file, "w") as f:
    json.dump(statistics, f, indent=4)

logger.info("=" * 60)
logger.info("Dataset Verification Finished")
logger.info(f"Manifest Saved : {manifest_file}")
logger.info("=" * 60)
logger.info("=" * 60)

logger.info(
    f"Total Corrupted Images : {len(corrupted_images)}"
)

logger.info("=" * 60)