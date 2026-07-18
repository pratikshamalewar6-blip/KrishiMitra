"""
KrishiMitra
YOLO Detection Configuration

Central configuration for
leaf detection.

Author:
    Pratiksha Malewar
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DetectionConfig:
    """
    YOLO Detection configuration.
    """

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    MODEL_NAME: str = "YOLOv11"

    MODEL_FILE: Path = (
        Path("saved_models")
        / "yolov11_leaf.pt"
    )

    # --------------------------------------------------
    # Image
    # --------------------------------------------------

    IMAGE_SIZE: int = 640

    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    CONFIDENCE_THRESHOLD: float = 0.35

    IOU_THRESHOLD: float = 0.45

    MAX_DETECTIONS: int = 10

    # --------------------------------------------------
    # Classes
    # --------------------------------------------------

    LEAF_CLASS_ID: int = 0

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    DEVICE: str = "cpu"

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------

    SAVE_RESULTS: bool = True

    SAVE_CROPS: bool = True

    SAVE_LABELS: bool = False

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    OUTPUT_DIRECTORY: Path = (
        Path("outputs")
        / "detections"
    )