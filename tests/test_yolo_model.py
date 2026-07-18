import shutil
from pathlib import Path
from ultralytics import YOLO

from detection.model import YOLOModel
from detection.config import DetectionConfig


def main():

    config = DetectionConfig()

    # Ensure the model file exists for testing
    model_path = Path(config.MODEL_FILE)
    if not model_path.exists():
        print(f"Model file {model_path} not found.")
        print("Downloading baseline yolo11n.pt model...")
        
        # Download pretrained weights using YOLO
        tmp_model = YOLO("yolo11n.pt")
        
        # Create parent directories
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy to config location
        shutil.copy("yolo11n.pt", model_path)
        print(f"Saved baseline model to {model_path}")

    model = YOLOModel(
        config.MODEL_FILE
    )

    model.load()

    print("=" * 60)
    print("YOLO Loaded Successfully")
    print("=" * 60)


if __name__ == "__main__":
    main()