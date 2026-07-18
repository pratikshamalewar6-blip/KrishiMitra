"""
KrishiMitra - YOLOv11 Leaf Detector Evaluation Script

Evaluates the trained YOLOv11 model on the processed leaf dataset.

Author:
    Pratiksha Malewar
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

from common.config import ConfigManager
from common.logger import LoggerManager
from common.file_utils import FileUtils

from detection.utils import prepare_leaf_dataset
from detection.config import DetectionConfig

logger = LoggerManager.get_logger("EvaluateDetector")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv11 Leaf Detector")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run evaluation, e.g., 'cpu', 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Dataset split to evaluate on (default: val)",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(f"Starting YOLOv11 Leaf Detector Evaluation ({args.split} split)")
    logger.info("=" * 60)

    # Load configuration
    config_mgr = ConfigManager()
    paths_config = config_mgr.get("paths")
    
    src_dataset_dir = Path(paths_config["datasets"]["plantdoc_detection"])
    processed_root = Path(paths_config["datasets"]["processed"])
    dest_dataset_dir = processed_root / "plantdoc_leaf"
    dataset_yaml_path = dest_dataset_dir / "dataset_leaf.yaml"

    # 1. Ensure dataset exists
    if not dataset_yaml_path.exists():
        logger.info("Leaf dataset not found. Preparing dataset now...")
        dataset_yaml_path = prepare_leaf_dataset(src_dataset_dir, dest_dataset_dir)

    # 2. Ensure model exists
    model_config = DetectionConfig()
    model_path = Path(model_config.MODEL_FILE)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Please run train_detector.py first."
        )

    # 3. Load YOLO Model
    logger.info(f"Loading trained model: {model_path}")
    model = YOLO(model_path)

    # 4. Run Evaluation
    logger.info(f"Running validation on split: {args.split}")
    metrics = model.val(
        data=str(dataset_yaml_path.resolve()).replace("\\", "/"),
        device=args.device,
        split=args.split,
        verbose=False,
    )

    # 5. Extract and Log Metrics
    # Ultralytics metrics mapping
    map50 = float(metrics.results_dict["metrics/mAP50(B)"])
    map50_95 = float(metrics.results_dict["metrics/mAP50-95(B)"])
    precision = float(metrics.results_dict["metrics/precision(B)"])
    recall = float(metrics.results_dict["metrics/recall(B)"])

    logger.info("-" * 60)
    logger.info("Evaluation Summary:")
    logger.info(f"mAP@50    : {map50:.4f}")
    logger.info(f"mAP@50-95 : {map50_95:.4f}")
    logger.info(f"Precision : {precision:.4f}")
    logger.info(f"Recall    : {recall:.4f}")
    logger.info("-" * 60)

    # 6. Save Evaluation Report
    report = {
        "dataset": "plantdoc_leaf",
        "split": args.split,
        "model_file": str(model_path),
        "metrics": {
            "mAP50": map50,
            "mAP50-95": map50_95,
            "precision": precision,
            "recall": recall,
        },
    }

    output_dir = Path(paths_config["outputs"]) / "detections"
    FileUtils.ensure_directory(output_dir)
    report_file = output_dir / f"evaluation_report_{args.split}.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logger.info(f"Evaluation report saved to: {report_file}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
