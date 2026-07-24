"""
KrishiMitra
SAM2 Model Loader

Handles downloading and loading of Segment Anything Model 2.

Author:
    Antigravity AI
"""

from __future__ import annotations

from pathlib import Path
from ultralytics import SAM
from common.logger import LoggerManager
from segmentation.config import SegmentationConfig


class SAM2ModelLoader:
    """
    Handles downloading and loading of the SAM2 model.
    """

    @staticmethod
    def load_model(config: SegmentationConfig) -> SAM:
        """
        Load or download and load the SAM2 model based on configuration.
        """
        logger = LoggerManager.get_logger("SAM2ModelLoader")
        
        # Check if saved_models directory exists
        saved_models_dir = Path("saved_models")
        saved_models_dir.mkdir(exist_ok=True)
        
        model_path = saved_models_dir / config.MODEL_FILE
        
        logger.info(f"Loading SAM2 model weights from: {model_path}")
        
        try:
            # Ultralytics downloads the model automatically if it doesn't exist
            model = SAM(str(model_path))
            logger.info("SAM2 model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load SAM2 model from '{model_path}': {e}")
            logger.info(f"Retrying fallback using default weights '{config.MODEL_FILE}'...")
            try:
                model = SAM(config.MODEL_FILE)
                logger.info("SAM2 fallback model loaded successfully.")
                return model
            except Exception as retry_err:
                logger.error(f"Fallback model loading failed: {retry_err}")
                raise retry_err
