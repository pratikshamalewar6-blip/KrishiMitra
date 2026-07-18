"""
KrishiMitra - YOLOv11 Leaf Detector Training Script

Prepares the single-class leaf dataset and trains a YOLOv11 model.

Author:
    Pratiksha Malewar
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# pyrefly: ignore [missing-import]
from ultralytics import YOLO

from common.config import ConfigManager
from common.logger import LoggerManager
from common.file_utils import FileUtils

from detection.utils import prepare_leaf_dataset
from detection.config import DetectionConfig

logger = LoggerManager.get_logger("TrainDetector")


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train YOLOv11 Leaf Detector")
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1 for quick run on CPU)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to train on, e.g., 'cpu', 'cuda', '0' (default: cpu)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting YOLOv11 Leaf Detector Training Process")
    logger.info("=" * 60)

    # Load configuration
    config_mgr = ConfigManager()
    paths_config = config_mgr.get("paths")
    
    src_dataset_dir = Path(paths_config["datasets"]["plantdoc_detection"])
    processed_root = Path(paths_config["datasets"]["processed"])
    dest_dataset_dir = processed_root / "plantdoc_leaf"

    # 1. Prepare Leaf Dataset
    dataset_yaml_path = prepare_leaf_dataset(src_dataset_dir, dest_dataset_dir)

    # 2. Load Base YOLO Model
    # Since config says yolo11n.pt, we check if it is downloaded or download it.
    model_name = config_mgr.get("detection.model_name", "yolo11n.pt")
    logger.info(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    # 3. Train YOLOv11
    logger.info(f"Training on device: {args.device} for {args.epochs} epoch(s)...")
    
    # Save results under outputs/detections/train
    project_dir = Path(config_mgr.get("paths.outputs", "outputs")) / "detections" / "train"
    
    results = model.train(
        data=str(dataset_yaml_path.resolve()).replace("\\", "/"),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=str(project_dir),
        name="leaf_detector",
        exist_ok=True,
    )

    # 4. Copy best weights to saved_models/yolov11_leaf.pt
    best_weights_path = project_dir / "leaf_detector" / "weights" / "best.pt"
    
    if best_weights_path.exists():
        target_model_file = Path(config_mgr.get("paths.saved_models", "saved_models")) / "yolov11_leaf.pt"
        target_model_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(best_weights_path, target_model_file)
        logger.info(f"Successfully trained model and saved weights to: {target_model_file}")
    else:
        logger.error(f"Could not find trained weights at: {best_weights_path}")

    logger.info("=" * 60)
    logger.info("YOLOv11 Leaf Detector Training Completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
