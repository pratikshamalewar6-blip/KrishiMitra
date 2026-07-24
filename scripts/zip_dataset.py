"""
KrishiMitra - Dataset Zip Helper

Compresses the processed single-class leaf dataset into a zip archive
for easy download and upload to Google Colab.

Author:
    Antigravity AI
"""

from __future__ import annotations

import shutil
from pathlib import Path
from common.logger import LoggerManager


def zip_dataset() -> None:
    logger = LoggerManager.get_logger("ZipDataset")
    
    # Path of the processed dataset
    source_dir = Path("datasets/processed/plantdoc_leaf")
    output_zip = Path("datasets/processed/plantdoc_leaf")
    
    if not source_dir.exists():
        logger.error(f"Source directory '{source_dir}' does not exist! Please run the dataset preprocessing first.")
        return
        
    logger.info(f"Zipping '{source_dir}' to '{output_zip}.zip' (this may take a minute)...")
    
    try:
        # Zip the directory
        shutil.make_archive(str(output_zip), "zip", str(source_dir))
        zip_file_path = output_zip.with_suffix(".zip")
        file_size_mb = zip_file_path.stat().st_size / (1024 * 1024)
        logger.info(f"Successfully zipped dataset to '{zip_file_path}' ({file_size_mb:.2f} MB)")
    except Exception as e:
        logger.error(f"Failed to zip dataset: {e}")


if __name__ == "__main__":
    zip_dataset()
