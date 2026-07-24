"""
KrishiMitra - SAM2 Leaf Segmentation CLI

Runs leaf detection followed by leaf segmentation to output background-removed leaf crops.

Author:
    Antigravity AI
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image

from common.logger import LoggerManager
from common.file_utils import FileUtils

from detection.detector import LeafDetector
from segmentation.segmenter import LeafSegmenter

logger = LoggerManager.get_logger("PredictSegmentation")


def process_image(
    detector: LeafDetector,
    segmenter: LeafSegmenter,
    image_path: Path,
) -> None:
    """
    Process a single image: detect leaves and segment them.
    """
    logger.info(f"Processing image: {image_path}")

    # Step 1: Run Leaf Detection
    detections = detector.detect(image_path)
    logger.info(f"Found {len(detections)} leaf/leaves.")

    if len(detections) == 0:
        logger.info("Skipping segmentation because no leaves were detected.")
        return

    # Step 2: Set up output directory
    image_name = image_path.stem
    output_dir = segmenter.config.OUTPUT_DIRECTORY / image_name
    FileUtils.ensure_directory(output_dir)

    # Load original image
    src_image = Image.open(image_path).convert("RGB")

    # Step 3: Run Segmentation for each detection
    extension = segmenter.config.OUTPUT_FORMAT.lower()
    
    for idx, detection in enumerate(detections):
        try:
            logger.info(f"Segmenting leaf {idx + 1} / {len(detections)}...")
            
            # Segment the leaf and remove background
            segmented_leaf = segmenter.segment_leaf(src_image, detection)
            
            # Save the crop
            output_file = output_dir / f"leaf_segmented_{idx + 1}.{extension}"
            
            if extension == "png":
                segmented_leaf.save(output_file, "PNG")
            else:
                segmented_leaf.save(output_file, "JPEG")
                
            logger.info(f"Saved segmented crop: {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to segment leaf {idx + 1}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YOLOv11 + SAM2 Joint Leaf Segmentation")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input image file or directory of images",
    )
    args = parser.parse_args()

    # Initialize YOLO detector and SAM2 segmenter
    logger.info("Initializing models...")
    detector = LeafDetector()
    segmenter = LeafSegmenter()

    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source path does not exist: {source_path}")

    # Process source (single file or directory)
    if source_path.is_file():
        process_image(detector, segmenter, source_path)
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
                process_image(detector, segmenter, img_file)
            except Exception as e:
                logger.error(f"Failed to process {img_file}: {e}")

    logger.info("Leaf detection and segmentation completed.")


if __name__ == "__main__":
    main()
