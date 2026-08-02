"""
KrishiMitra - Master Sequential 3-Stage Training Pipeline

Orchestrates sequential model training:
Stage 1: PlantVillage Disease Classification (EfficientNet-B0)
Stage 2: PlantDoc Real-World Fine-Tuning (Transfer Learning)
Stage 3: PlantDoc Leaf Detection (YOLOv11 Detector)

Author:
    Antigravity AI
"""

from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

# Add project root to sys.path
pkg_root = Path(__file__).resolve().parent.parent
if str(pkg_root) not in sys.path:
    sys.path.append(str(pkg_root))

from common.logger import LoggerManager
logger = LoggerManager.get_logger("TrainFullPipeline")


def run_stage_1(epochs: int, batch_size: int, device: str) -> None:
    """Stage 1: Train EfficientNet-B0 on PlantVillage Dataset."""
    logger.info("\n" + "=" * 70)
    logger.info("  STAGE 1: PlantVillage Disease Classification Training")
    logger.info("=" * 70)
    
    cmd = [
        sys.executable,
        "classification/train_classifier.py",
        "--dataset", "plantvillage",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--device", device
    ]
    logger.info(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=pkg_root, check=True)


def run_stage_2(epochs: int, stage1_epochs: int, batch_size: int, device: str) -> None:
    """Stage 2: Fine-Tune on PlantDoc / Merged Dataset from Stage 1 Checkpoint."""
    logger.info("\n" + "=" * 70)
    logger.info("  STAGE 2: PlantDoc Real-World Disease Classification Fine-Tuning")
    logger.info("=" * 70)
    
    cmd = [
        sys.executable,
        "classification/train_realworld.py",
        "--epochs", str(epochs),
        "--stage1-epochs", str(stage1_epochs),
        "--batch-size", str(batch_size),
        "--pretrained-path", "saved_models/efficientnet_b0_disease.pt",
        "--device", device
    ]
    logger.info(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=pkg_root, check=True)


def run_stage_3(epochs: int, batch_size: int, device: str) -> None:
    """Stage 3: Train YOLOv11 Leaf Detector on PlantDoc Detection Dataset."""
    logger.info("\n" + "=" * 70)
    logger.info("  STAGE 3: PlantDoc YOLOv11 Leaf Detector Training")
    logger.info("=" * 70)
    
    cmd = [
        sys.executable,
        "detection/train_detector.py",
        "--epochs", str(epochs),
        "--batch", str(batch_size),
        "--device", device
    ]
    logger.info(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=pkg_root, check=True)


def main():
    parser = argparse.ArgumentParser(description="Master 3-Stage Training Pipeline for KrishiMitra AI Models")
    parser.add_argument("--stage", type=str, default="all", choices=["all", "1", "2", "3"], help="Stage to run ('all', '1', '2', '3')")
    parser.add_argument("--epochs-stage1", type=int, default=10, help="Epochs for Stage 1 (PlantVillage)")
    parser.add_argument("--epochs-stage2", type=int, default=15, help="Epochs for Stage 2 (PlantDoc Fine-Tuning)")
    parser.add_argument("--stage1-head-epochs", type=int, default=3, help="Head freeze epochs for Stage 2")
    parser.add_argument("--epochs-stage3", type=int, default=10, help="Epochs for Stage 3 (YOLO Leaf Detection)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--device", type=str, default="cuda", help="Target device ('cuda', 'cpu', etc.)")
    args = parser.parse_args()

    logger.info("Starting Master KrishiMitra AI Retraining Pipeline...")
    logger.info(f"Selected Stage : {args.stage}")
    logger.info(f"Target Device  : {args.device}")

    try:
        if args.stage in ("all", "1"):
            run_stage_1(epochs=args.epochs_stage1, batch_size=args.batch_size, device=args.device)

        if args.stage in ("all", "2"):
            run_stage_2(epochs=args.epochs_stage2, stage1_epochs=args.stage1_head_epochs, batch_size=args.batch_size, device=args.device)

        if args.stage in ("all", "3"):
            run_stage_3(epochs=args.epochs_stage3, batch_size=args.batch_size, device=args.device)

        logger.info("\n" + "=" * 70)
        logger.info("🎉 Master Sequential Training Pipeline Completed Successfully!")
        logger.info("Saved Checkpoints:")
        logger.info(" - Stage 1 Base Weights      : saved_models/efficientnet_b0_disease.pt")
        logger.info(" - Stage 2 RealWorld Weights : saved_models/efficientnet_b0_realworld.pt")
        logger.info(" - Stage 3 YOLO Leaf Detector: saved_models/yolov11_leaf.pt")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"Training pipeline encountered an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
