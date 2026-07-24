"""
KrishiMitra
EfficientNet-B0 Disease Classification Configuration

Central configuration for disease classification.

Author:
    Antigravity AI
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import torch
from common.config import ConfigManager


@dataclass
class ClassificationConfig:
    """
    Disease Classification configuration.
    """

    ARCHITECTURE: str = "efficientnet_b0"
    PRETRAINED: bool = True
    INPUT_SIZE: int = 224
    NUM_CLASSES: int = 38

    # Training settings
    RANDOM_SEED: int = 42
    EPOCHS: int = 30
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 2
    LEARNING_RATE: float = 0.0001
    WEIGHT_DECAY: float = 0.0001
    VALIDATION_SPLIT: float = 0.15
    TEST_SPLIT: float = 0.15

    # Early stopping & checkpoint
    EARLY_STOPPING_ENABLED: bool = True
    EARLY_STOPPING_PATIENCE: int = 7
    EARLY_STOPPING_MONITOR: str = "validation_loss"
    CHECKPOINT_MONITOR: str = "validation_accuracy"

    # Optimizer & Scheduler
    OPTIMIZER_NAME: str = "AdamW"
    SCHEDULER_NAME: str = "CosineAnnealingLR"

    # Directory Paths
    MODEL_FILE: Path = field(default_factory=lambda: Path("saved_models") / "efficientnet_b0_disease.pt")
    OUTPUT_DIRECTORY: Path = field(default_factory=lambda: Path("outputs") / "classification")
    TENSORBOARD_DIRECTORY: Path = field(default_factory=lambda: Path("logs") / "tensorboard")

    # Device
    DEVICE: str = "cpu"

    # Augmentation
    HORIZONTAL_FLIP: bool = True
    VERTICAL_FLIP: bool = False
    ROTATION: int = 20
    BRIGHTNESS: float = 0.2
    CONTRAST: float = 0.2
    BLUR: bool = True

    def __post_init__(self) -> None:
        try:
            config = ConfigManager()
            
            # Model configs
            self.ARCHITECTURE = config.get("classification.architecture", "efficientnet_b0")
            self.PRETRAINED = config.get("classification.pretrained", True)
            self.INPUT_SIZE = config.get("classification.input_size", 224)
            self.NUM_CLASSES = config.get("classification.num_classes", 38)
            
            # Training configs
            self.RANDOM_SEED = config.get("training.random_seed", 42)
            self.EPOCHS = config.get("training.epochs", 30)
            self.BATCH_SIZE = config.get("training.batch_size", 32)
            self.NUM_WORKERS = config.get("training.num_workers", 2)
            self.LEARNING_RATE = config.get("training.learning_rate", 0.0001)
            self.WEIGHT_DECAY = config.get("training.weight_decay", 0.0001)
            self.VALIDATION_SPLIT = config.get("training.validation_split", 0.15)
            self.TEST_SPLIT = config.get("training.test_split", 0.15)
            
            # Early stopping & checkpoint
            self.EARLY_STOPPING_ENABLED = config.get("early_stopping.enabled", True)
            self.EARLY_STOPPING_PATIENCE = config.get("early_stopping.patience", 7)
            self.EARLY_STOPPING_MONITOR = config.get("early_stopping.monitor", "validation_loss")
            self.CHECKPOINT_MONITOR = config.get("checkpoint.monitor", "validation_accuracy")
            
            # Optimizer & Scheduler
            self.OPTIMIZER_NAME = config.get("optimizer.name", "AdamW")
            self.SCHEDULER_NAME = config.get("scheduler.name", "CosineAnnealingLR")
            
            # Paths
            saved_models_root = config.get("paths.saved_models", "saved_models")
            self.MODEL_FILE = Path(saved_models_root) / "efficientnet_b0_disease.pt"
            
            outputs_root = config.get("paths.outputs", "outputs")
            self.OUTPUT_DIRECTORY = Path(outputs_root) / "classification"
            
            tb_root = config.get("paths.tensorboard", "logs/tensorboard")
            self.TENSORBOARD_DIRECTORY = Path(tb_root)
            
            # Augmentations
            self.HORIZONTAL_FLIP = config.get("augmentation.horizontal_flip", True)
            self.VERTICAL_FLIP = config.get("augmentation.vertical_flip", False)
            self.ROTATION = config.get("augmentation.rotation", 20)
            self.BRIGHTNESS = config.get("augmentation.brightness", 0.2)
            self.CONTRAST = config.get("augmentation.contrast", 0.2)
            self.BLUR = config.get("augmentation.blur", True)
            
            # Device configuration
            yaml_device = config.get("training.device", "auto")
            if yaml_device == "auto" or yaml_device == "gpu":
                use_gpu = config.get("device.use_gpu", True)
                self.DEVICE = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
            else:
                self.DEVICE = yaml_device
                
        except Exception:
            # Fallback gracefully to defaults if configs aren't available
            pass
