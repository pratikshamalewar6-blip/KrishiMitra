"""
KrishiMitra - YOLOv11 Leaf Detector Model Export Script

Exports the trained PyTorch YOLOv11 model (.pt) to deployment formats like ONNX.

Author:
    Pratiksha Malewar
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

from common.logger import LoggerManager
from detection.config import DetectionConfig

logger = LoggerManager.get_logger("ExportDetector")


def main():
    parser = argparse.ArgumentParser(description="Export YOLOv11 Model")
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "tflite", "openvino", "engine"],
        help="Target export format (default: onnx)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Starting YOLOv11 Model Export (Format: {args.format})")
    logger.info("=" * 60)

    # 1. Load model configuration
    config = DetectionConfig()
    model_path = Path(config.MODEL_FILE)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please run train_detector.py first."
        )

    # 2. Load model
    logger.info(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # 3. Export model
    logger.info(f"Exporting model to format: {args.format}...")
    try:
        exported_path = model.export(format=args.format)
        logger.info(f"Model successfully exported to: {exported_path}")
    except Exception as e:
        logger.error(f"Failed to export model: {e}")
        raise e

    logger.info("=" * 60)
    logger.info("YOLOv11 Model Export Process Finished")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
