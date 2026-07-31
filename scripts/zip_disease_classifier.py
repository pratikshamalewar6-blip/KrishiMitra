"""
KrishiMitra - Disease Classifier Zip Helper

Prepares and compresses:
1. The PlantVillage dataset (datasets/raw/plantvillage) into a zip file.
2. The codebase (excluding datasets, model weights, venv, git, and logs) into a lightweight zip file.

For easy download and upload to Google Colab.

Author:
    Antigravity AI
"""

from __future__ import annotations

import sys
import shutil
import zipfile
from pathlib import Path

# Automatically add parent directory (disease_detection) to sys.path
script_dir = Path(__file__).resolve().parent
disease_detection_dir = script_dir.parent
if str(disease_detection_dir) not in sys.path:
    sys.path.insert(0, str(disease_detection_dir))

from common.logger import LoggerManager

logger = LoggerManager.get_logger("ZipDiseaseClassifier")


def zip_dataset() -> Path | None:
    source_dir = Path("datasets/raw/plantvillage")
    output_zip_path = Path("datasets/raw/plantvillage_dataset.zip")
    
    if not source_dir.exists():
        logger.error(f"Source dataset directory '{source_dir}' does not exist!")
        return None
        
    logger.info(f"Zipping dataset '{source_dir}' to '{output_zip_path}' file-by-file...")
    try:
        # Delete old zip if exists
        if output_zip_path.exists():
            try:
                output_zip_path.unlink()
            except Exception as delete_err:
                logger.warning(f"Could not delete old zip file: {delete_err}")
            
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            count = 0
            for path in source_dir.rglob("*"):
                if path.is_file():
                    # Write relative to source_dir
                    arcname = path.relative_to(source_dir)
                    zipf.write(path, arcname)
                    count += 1
                    if count % 10000 == 0:
                        logger.info(f"Zipped {count} dataset files...")
                        
        file_size_mb = output_zip_path.stat().st_size / (1024 * 1024)
        logger.info(f"Successfully zipped {count} dataset files to '{output_zip_path}' ({file_size_mb:.2f} MB)")
        return output_zip_path
    except Exception as e:
        logger.error(f"Failed to zip dataset: {e}")
        return None


def zip_codebase() -> Path | None:
    output_zip_path = Path("outputs/disease_detection_code.zip")
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Zipping codebase to '{output_zip_path}'...")
    
    # Exclude directories
    exclude_dirs = {
        ".git",
        "venv",
        "datasets",
        "saved_models",
        "logs",
        "runs",
        "outputs",
        "__pycache__",
        ".ipynb_checkpoints",
        ".vscode",
    }
    
    # Exclude file extensions
    exclude_exts = {".pt", ".onnx", ".pyc", ".zip", ".log"}
    
    try:
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            count = 0
            for path in Path(".").glob("**/*"):
                # Check if it should be excluded
                parts = path.parts
                if any(part in exclude_dirs for part in parts):
                    continue
                if path.is_file():
                    if path.suffix in exclude_exts:
                        continue
                    zipf.write(path, path)
                    count += 1
                    
            # Explicitly include the splits folder containing CSV files (since outputs was excluded)
            splits_dir = Path("outputs/splits")
            if splits_dir.exists():
                for path in splits_dir.glob("*.csv"):
                    zipf.write(path, Path("outputs/splits") / path.name)
                    count += 1

            # Explicitly include classification mapping JSON files
            classification_dir = Path("outputs/classification")
            if classification_dir.exists():
                for path in classification_dir.glob("*.json"):
                    zipf.write(path, Path("outputs/classification") / path.name)
                    count += 1
                    
            # Explicitly include verify_merged_dataset.py if present at root
            verify_script = Path("verify_merged_dataset.py")
            if verify_script.exists():
                zipf.write(verify_script, verify_script)
                count += 1
                    
        file_size_kb = output_zip_path.stat().st_size / 1024
        logger.info(f"Successfully zipped {count} code files to '{output_zip_path}' ({file_size_kb:.2f} KB)")
        return output_zip_path
    except Exception as e:
        logger.error(f"Failed to zip codebase: {e}")
        return None


def zip_plantdoc_crops() -> Path | None:
    output_zip_path = Path("outputs/plantdoc_crops.zip")
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    
    candidate_sources = [
        Path("datasets/processed/plantdoc_classification"),
        Path("datasets/raw/plantdoc_classification")
    ]
    
    valid_sources = [d for d in candidate_sources if d.exists()]
    if not valid_sources:
        logger.warning("No PlantDoc directory found. Skipping crops zipping.")
        return None
        
    logger.info(f"Zipping segmented crops from {valid_sources} to '{output_zip_path}'...")
    try:
        if output_zip_path.exists():
            try:
                output_zip_path.unlink()
            except Exception as delete_err:
                logger.warning(f"Could not delete old zip file: {delete_err}")
                
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            count = 0
            for src_dir in valid_sources:
                for path in src_dir.rglob("*"):
                    if path.is_file():
                        arcname = path.relative_to(src_dir)
                        zipf.write(path, src_dir / arcname)
                        count += 1
                        
        file_size_mb = output_zip_path.stat().st_size / (1024 * 1024)
        logger.info(f"Successfully zipped {count} crop images to '{output_zip_path}' ({file_size_mb:.2f} MB)")
        return output_zip_path
    except Exception as e:
        logger.error(f"Failed to zip segmented crops: {e}")
        return None


def main() -> None:
    logger.info("=" * 60)
    logger.info("Starting Disease Classifier Packaging Process")
    logger.info("=" * 60)
    
    zip_codebase()
    zip_plantdoc_crops()
    logger.info("Note: PlantVillage dataset zipping skipped. It will be downloaded directly in Google Colab.")
    
    logger.info("=" * 60)
    logger.info("Packaging Process Finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
