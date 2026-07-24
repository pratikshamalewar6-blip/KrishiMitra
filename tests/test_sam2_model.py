"""
KrishiMitra - SAM2 Leaf Segmenter Unit Test

Tests model initialization and prediction on a mock image.

Author:
    Antigravity AI
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from segmentation.config import SegmentationConfig
from segmentation.segmenter import LeafSegmenter


def test_sam2_loading_and_prediction() -> None:
    print("=" * 60)
    print("Testing SAM2 Model Loading & Segmenter")
    print("=" * 60)
    
    # 1. Initialize configuration and segmenter
    config = SegmentationConfig()
    segmenter = LeafSegmenter(config)
    print(f"LeafSegmenter initialized with weights: {config.MODEL_FILE}")
    
    # 2. Create a dummy image (H=400, W=400)
    # Draw a green square representing the leaf object on a black background
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[120:280, 120:280] = [100, 255, 100]  # Green square leaf
    
    # Bounding box for the object: x1=110, y1=110, x2=290, y2=290
    bbox = [110, 110, 290, 290]
    
    print("Running dummy segment_leaf prediction...")
    segmented_crop = segmenter.segment_leaf(img, bbox)
    
    print(f"Segmented crop size: {segmented_crop.size} (Mode: {segmented_crop.mode})")
    
    # Verify outputs
    assert segmented_crop is not None
    assert isinstance(segmented_crop, Image.Image)
    
    if config.OUTPUT_FORMAT.lower() == "png":
        assert segmented_crop.mode == "RGBA", f"Expected RGBA output, got {segmented_crop.mode}"
    else:
        assert segmented_crop.mode == "RGB", f"Expected RGB output, got {segmented_crop.mode}"
        
    print("=" * 60)
    print("SAM2 Segmenter Test Passed Successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_sam2_loading_and_prediction()
