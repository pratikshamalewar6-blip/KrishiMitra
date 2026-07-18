"""
KrishiMitra - YOLOv11 Model Loader

Loads the trained YOLOv11 model.

Author:
    Pratiksha Malewar

Project:
    KrishiMitra
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from common.logger import LoggerManager


class YOLOModel:
    """
    Singleton wrapper around YOLO model.

    Loads the model only once.
    """

    _model = None

    def __init__(
        self,
        model_path: str | Path,
    ) -> None:

        self.logger = LoggerManager.get_logger(
            "YOLOModel"
        )

        self.model_path = Path(model_path)

    # ------------------------------------------------------

    def load(self) -> YOLO:
        """
        Load YOLO model.

        Returns
        -------
        YOLO
        """

        if YOLOModel._model is not None:
            return YOLOModel._model

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.logger.info(
            f"Loading YOLO model: {self.model_path}"
        )

        YOLOModel._model = YOLO(
            str(self.model_path)
        )

        self.logger.info(
            "YOLO model loaded successfully."
        )

        return YOLOModel._model

    # ------------------------------------------------------

    @property
    def model(self) -> YOLO:
        """
        Returns loaded model.
        """

        return self.load()