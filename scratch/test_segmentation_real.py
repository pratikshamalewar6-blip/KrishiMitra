"""
KrishiMitra - Real Image SAM2 Segmentation Test

Converts ground-truth YOLO bounding box to pixel coordinates and runs SAM2 leaf segmentation.

Author:
    Antigravity AI
"""

from __future__ import annotations

from pathlib import Path
from PIL import Image
import numpy as np

from segmentation.config import SegmentationConfig
from segmentation.segmenter import LeafSegmenter


def main() -> None:
    print("=" * 60)
    print("Testing SAM2 Leaf Segmentation on a Real Image")
    print("=" * 60)

    # 1. Load image
    image_path = Path("datasets/raw/plantdoc_detection/images/val/val_00000.jpg")
    if not image_path.exists():
        print(f"Error: Image not found at {image_path}")
        return

    img = Image.open(image_path)
    W, H = img.size
    print(f"Loaded image: {image_path} (Size: {W}x{H})")

    # 2. Parse YOLO normalized bbox: class_id x_center y_center width height
    # Box 1: 25 0.354914 0.492318 0.695855 0.826816
    x_center, y_center, w, h = 0.354914, 0.492318, 0.695855, 0.826816

    # Convert to absolute pixel coordinates [x1, y1, x2, y2]
    x1 = int((x_center - w / 2) * W)
    y1 = int((y_center - h / 2) * H)
    x2 = int((x_center + w / 2) * W)
    y2 = int((y_center + h / 2) * H)

    # Bound coordinates to image dimensions
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)

    bbox = [x1, y1, x2, y2]
    print(f"Converted YOLO bounding box to pixels: {bbox}")

    # 3. Load segmenter and run prediction
    config = SegmentationConfig()
    segmenter = LeafSegmenter(config)
    
    print("Running SAM2 segmentation...")
    segmented_leaf = segmenter.segment_leaf(img, bbox)

    # 4. Save results
    output_dir = Path("outputs/segmentations")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "val_00000_mock_segmented.png"
    
    segmented_leaf.save(output_path, "PNG")
    print(f"Successfully saved background-removed crop to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
