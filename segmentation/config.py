"""
KrishiMitra
SAM2 Leaf Segmentation Configuration

Central configuration for leaf segmentation.

Author:
    Antigravity AI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from common.config import ConfigManager


@dataclass
class SegmentationConfig:
    """
    SAM2 Segmentation configuration.
    """

    MODEL_NAME: str = "SAM2"

    # Default weights file name for Ultralytics SAM2 Hiera Tiny
    MODEL_FILE: str = "sam2_t.pt"

    REMOVE_BACKGROUND: bool = True

    OUTPUT_FORMAT: str = "png"

    OUTPUT_DIRECTORY: Path = field(default_factory=lambda: Path("outputs") / "segmentations")

    DEVICE: str = "cpu"

    def __post_init__(self) -> None:
        try:
            config = ConfigManager()
            
            yaml_model_name = config.get("segmentation.model_name", "sam2_hiera_tiny")
            if "tiny" in yaml_model_name or "t" in yaml_model_name:
                self.MODEL_FILE = "sam2_t.pt"
            elif "small" in yaml_model_name or "s" in yaml_model_name:
                self.MODEL_FILE = "sam2_s.pt"
            elif "base" in yaml_model_name or "b" in yaml_model_name:
                self.MODEL_FILE = "sam2_b.pt"
            elif "large" in yaml_model_name or "l" in yaml_model_name:
                self.MODEL_FILE = "sam2_l.pt"
            
            self.REMOVE_BACKGROUND = config.get("segmentation.remove_background", True)
            self.OUTPUT_FORMAT = config.get("segmentation.output_format", "png")
            
            # Read output paths if present
            outputs_root = config.get("paths.outputs", "outputs")
            self.OUTPUT_DIRECTORY = Path(outputs_root) / "segmentations"
            
        except Exception:
            # Fallback gracefully to default class attribute assignments
            pass
