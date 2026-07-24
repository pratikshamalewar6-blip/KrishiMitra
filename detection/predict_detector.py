"""
KrishiMitra - YOLOv11 Leaf Detector Prediction CLI

Runs leaf detection on images or directories and saves annotated results and crops.

Author:
    Pratiksha Malewar
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path


from PIL import Image

from common.logger import LoggerManager
from common.file_utils import FileUtils

from detection.detector import LeafDetector
from detection.utils import visualize_detections

logger = LoggerManager.get_logger("PredictDetector")


def process_image(
    detector: LeafDetector,
    image_path: Path,
    save_crops: bool,
    save_viz: bool,
) -> None:
    """
    Process a single image for leaf detection.
    """
    logger.info(f"Processing image: {image_path}")

    # Run detection
    detections = detector.detect(image_path)
    logger.info(f"Found {len(detections)} leaf/leaves.")

    image_name = image_path.stem

    # Save visual annotations
    if save_viz:
        output_viz_path = (
            detector.config.OUTPUT_DIRECTORY
            / f"{image_name}_annotated.jpg"
        )
        visualize_detections(image_path, detections, output_viz_path)

    # Save cropped leaves
    if save_crops and len(detections) > 0:
        crops = detector.crop_leaves(image_path, detections)
        detector.save_crops(image_path, crops)


def main():
    parser = argparse.ArgumentParser(description="Run YOLOv11 Leaf Detection Inference")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input image file or directory of images",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Confidence threshold override",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="IoU threshold override",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        default=True,
        help="Save cropped leaf images (default: True)",
    )
    parser.add_argument(
        "--no-save-crops",
        dest="save_crops",
        action="store_false",
        help="Do not save cropped leaf images",
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        default=True,
        help="Save annotated visualization images (default: True)",
    )
    parser.add_argument(
        "--no-save-viz",
        dest="save_viz",
        action="store_false",
        help="Do not save annotated visualization images",
    )
    args = parser.parse_args()

    # Initialize Detector
    detector = LeafDetector()

    # Apply configuration overrides if provided
    overrides = {}
    if args.conf is not None:
        overrides["CONFIDENCE_THRESHOLD"] = args.conf
    if args.iou is not None:
        overrides["IOU_THRESHOLD"] = args.iou
    
    if overrides:
        detector.config = replace(detector.config, **overrides)
        logger.info(f"Applied overrides to detector config: {overrides}")

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    # Process source (single file or directory)
    if source_path.is_file():
        process_image(detector, source_path, args.save_crops, args.save_viz)
    elif source_path.is_dir():
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_files = sorted([
            f for f in source_path.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ])

        if not image_files:
            logger.warning(f"No supported images found in directory: {source_path}")
            return

        logger.info(f"Processing {len(image_files)} images from directory...")
        for img_file in image_files:
            try:
                process_image(detector, img_file, args.save_crops, args.save_viz)
            except Exception as e:
                logger.error(f"Failed to process {img_file}: {e}")

    logger.info("Leaf detection inference completed.")


if __name__ == "__main__":
    main()
