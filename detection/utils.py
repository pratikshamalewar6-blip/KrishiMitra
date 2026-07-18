"""
KrishiMitra - YOLO Leaf Detection Utilities

Provides dataset preparation, class mapping, and image visualization utilities.

Author:
    Pratiksha Malewar
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List

import cv2
import yaml
from PIL import Image

from common.logger import LoggerManager
from common.file_utils import FileUtils

logger = LoggerManager.get_logger("DetectionUtils")


def prepare_leaf_dataset(
    src_dir: str | Path,
    dest_dir: str | Path,
) -> Path:
    """
    Preprocess PlantDoc object detection dataset to map all 29 classes
    to a single 'leaf' class (ID 0).

    Copies images and writes modified labels to the destination directory.
    Also creates the dataset_leaf.yaml configuration file.

    Parameters
    ----------
    src_dir : str | Path
        Root directory of the raw PlantDoc detection dataset.
    dest_dir : str | Path
        Destination directory to save the processed single-class dataset.

    Returns
    -------
    Path
        Path to the generated dataset_leaf.yaml configuration file.
    """
    src_dir = Path(src_dir)
    dest_dir = Path(dest_dir)

    logger.info(f"Preparing single-class leaf dataset from {src_dir} to {dest_dir}")

    # Create destination directories
    for split in ["train", "val"]:
        FileUtils.ensure_directory(dest_dir / "images" / split)
        FileUtils.ensure_directory(dest_dir / "labels" / split)

    # Process each split
    for split in ["train", "val"]:
        src_images_dir = src_dir / "images" / split
        src_labels_dir = src_dir / "labels" / split

        dest_images_dir = dest_dir / "images" / split
        dest_labels_dir = dest_dir / "labels" / split

        if not src_images_dir.exists():
            logger.warning(f"Source images directory not found: {src_images_dir}")
            continue

        # Get list of images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = [
            f for f in src_images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        logger.info(f"Processing {len(image_files)} images for {split} split...")

        for img_path in image_files:
            # 1. Copy image to destination
            dest_img_path = dest_images_dir / img_path.name
            if not dest_img_path.exists():
                shutil.copy(img_path, dest_img_path)

            # 2. Modify and write label to destination
            # YOLO label has the same stem as image but with .txt extension
            label_name = f"{img_path.stem}.txt"
            src_label_path = src_labels_dir / label_name
            dest_label_path = dest_labels_dir / label_name

            if src_label_path.exists():
                modified_lines = []
                with open(src_label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            # Map any original class ID to 0 (leaf)
                            parts[0] = "0"
                            modified_lines.append(" ".join(parts))

                with open(dest_label_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(modified_lines) + "\n")
            else:
                # If no label exists, create an empty label file (no detections)
                with open(dest_label_path, "w", encoding="utf-8") as f:
                    pass

    # Generate dataset_leaf.yaml
    dataset_yaml = dest_dir / "dataset_leaf.yaml"
    yaml_content = {
        "path": str(dest_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "leaf"},
    }

    with open(dataset_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, default_flow_style=False)

    logger.info(f"Dataset leaf configuration saved to: {dataset_yaml}")
    return dataset_yaml


def visualize_detections(
    image_path: str | Path,
    detections: List[dict] | List[any],
    output_path: str | Path,
) -> None:
    """
    Draw bounding boxes on the image and save the annotated result.

    Parameters
    ----------
    image_path : str | Path
        Path to the original image.
    detections : List[DetectionResult]
        List of detection result objects containing x1, y1, x2, y2, and confidence.
    output_path : str | Path
        Path to save the annotated image.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)

    # Read image using OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        logger.error(f"Could not read image for visualization: {image_path}")
        return

    # Draw boxes
    for det in detections:
        # Access attributes dynamically to support both DetectionResult objects
        # and dictionaries
        x1 = int(getattr(det, "x1", det.get("x1") if isinstance(det, dict) else 0))
        y1 = int(getattr(det, "y1", det.get("y1") if isinstance(det, dict) else 0))
        x2 = int(getattr(det, "x2", det.get("x2") if isinstance(det, dict) else 0))
        y2 = int(getattr(det, "y2", det.get("y2") if isinstance(det, dict) else 0))
        conf = float(getattr(det, "confidence", det.get("confidence") if isinstance(det, dict) else 0.0))

        # Box parameters
        color = (0, 255, 0)  # Green
        thickness = 2
        
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Label text
        text = f"Leaf: {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        # Text size and background
        (text_width, text_height), _ = cv2.getTextSize(
            text, font, font_scale, font_thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(
            img,
            (x1, y1 - text_height - 5),
            (x1 + text_width + 5, y1),
            color,
            -1,  # Filled
        )
        
        # Draw white text
        cv2.putText(
            img,
            text,
            (x1 + 2, y1 - 4),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    # Ensure output directory exists and save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)
    logger.info(f"Annotated image saved to: {output_path}")
