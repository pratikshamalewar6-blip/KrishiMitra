"""
KrishiMitra
YOLOv11 Leaf Detector

Loads YOLO model and detects leaves.

Author:
    Pratiksha Malewar

Project:
    KrishiMitra
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List

from PIL import Image

from ultralytics import YOLO

from common.logger import LoggerManager
from common.file_utils import FileUtils

from detection.config import DetectionConfig


# ==========================================================
# Detection Result
# ==========================================================

@dataclass
class DetectionResult:
    """
    Represents one detected leaf.
    """

    x1: int
    y1: int
    x2: int
    y2: int

    confidence: float

    class_id: int


# ==========================================================
# Leaf Detector
# ==========================================================

class LeafDetector:
    """
    YOLO Leaf Detector.
    """

    def __init__(self):

        self.logger = LoggerManager.get_logger(
            "LeafDetector"
        )

        self.config = DetectionConfig()

        FileUtils.ensure_directory(
            self.config.OUTPUT_DIRECTORY
        )

        self.model = self.load_model()

    # ------------------------------------------------------

    def load_model(
        self,
    ) -> YOLO:
        """
        Load YOLO model.
        """

        model_path = self.config.MODEL_FILE

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found: {model_path}"
            )

        self.logger.info(
            f"Loading YOLO model : {model_path}"
        )

        return YOLO(model_path)

    # ------------------------------------------------------

    def detect(
        self,
        image_path: str | Path,
    ) -> List[DetectionResult]:
        """
        Detect leaves in an image.
        """

        image_path = Path(image_path)

        if not image_path.exists():

            raise FileNotFoundError(image_path)

        results = self.model.predict(

            source=str(image_path),

            imgsz=self.config.IMAGE_SIZE,

            conf=self.config.CONFIDENCE_THRESHOLD,

            iou=self.config.IOU_THRESHOLD,

            max_det=self.config.MAX_DETECTIONS,

            device=self.config.DEVICE,

            verbose=False,
        )

        detections = []

        result = results[0]

        if result.boxes is None:

            return detections

        for box in result.boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            detections.append(

                DetectionResult(

                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,

                    confidence=confidence,

                    class_id=class_id,
                )
            )

        self.logger.info(
            f"Detected {len(detections)} leaf(s)."
        )

        return detections

    # ------------------------------------------------------

    def crop_leaves(
        self,
        image_path: str | Path,
        detections: List[DetectionResult],
    ) -> List[Image.Image]:
        """
        Crop detected leaves.
        """

        image = Image.open(
            image_path
        ).convert("RGB")

        crops = []

        for detection in detections:

            crop = image.crop(

                (
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2,
                )

            )

            crops.append(crop)

        return crops

    # ------------------------------------------------------

    def save_crops(
        self,
        image_path: str | Path,
        crops: List[Image.Image],
    ) -> None:
        """
        Save cropped leaves.
        """

        image_name = Path(
            image_path
        ).stem

        output_dir = (
            self.config.OUTPUT_DIRECTORY
            / image_name
        )

        FileUtils.ensure_directory(
            output_dir
        )

        for index, crop in enumerate(crops):

            crop.save(

                output_dir
                / f"leaf_{index+1}.jpg"

            )

        self.logger.info(

            f"Saved {len(crops)} crops."

        )


# ==========================================================
# Main
# ==========================================================

def main():

    detector = LeafDetector()

    image_path = (
        "sample.jpg"
    )

    detections = detector.detect(
        image_path
    )

    print("=" * 60)

    print(
        f"Leaves : {len(detections)}"
    )

    for detection in detections:

        print(detection)

    crops = detector.crop_leaves(
        image_path,
        detections,
    )

    detector.save_crops(
        image_path,
        crops,
    )


if __name__ == "__main__":

    main()