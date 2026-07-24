"""
KrishiMitra - Phase 4.6: Real-World Pipeline Validation & Error Analysis

Executes the full pipeline (YOLOv11 -> SAM2 -> EfficientNet-B0) on real-world images.
Measures final end-to-end pipeline accuracy and saves incorrect classifications
to 'analysis/failed_cases/' for error analysis.

Author:
    Antigravity AI
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import torch

from common.logger import LoggerManager
from common.file_utils import FileUtils
from detection.detector import LeafDetector
from segmentation.segmenter import LeafSegmenter
from segmentation.config import SegmentationConfig
from classification.config import ClassificationConfig
from classification.model import DiseaseClassifier
from classification.predict_classifier import predict_disease
from data.preprocess_plantdoc import parse_ground_truth, calculate_iou

logger = LoggerManager.get_logger("RealWorldValidation")


def draw_error_overlay(
    image: Image.Image,
    box: tuple[int, int, int, int],
    true_label: str,
    pred_label: str,
    confidence: float
) -> Image.Image:
    """Draws a red error bounding box and text overlay on the image."""
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    # Draw red rectangle
    draw.rectangle(box, outline="red", width=4)
    
    # Draw text background
    x1, y1, x2, y2 = box
    text = f"True: {true_label} | Pred: {pred_label} ({confidence*100:.1f}%)"
    
    # Simple text drawing
    draw.rectangle([x1, y1 - 25, x1 + len(text) * 7, y1], fill="red")
    draw.text((x1 + 5, y1 - 20), text, fill="white")
    
    return img_draw


def main() -> None:
    parser = argparse.ArgumentParser(description="Run End-to-End Real-World Validation Pipeline")
    parser.add_argument("--source", type=str, default="datasets/raw/plantdoc_detection/images/val", help="Path to real-world validation images")
    parser.add_argument("--labels", type=str, default="datasets/raw/plantdoc_detection/labels/val", help="Path to ground-truth label files")
    parser.add_argument("--weights", type=str, default="saved_models/efficientnet_b0_realworld.pt", help="Path to classification weights")
    parser.add_argument("--threshold", type=float, default=0.60, help="OOD rejection threshold")
    parser.add_argument("--limit", type=int, default=100, help="Limit number of validation images to run")
    args = parser.parse_args()

    # 1. Initialize Pipeline Components
    logger.info("Initializing complete computer vision pipeline...")
    detector = LeafDetector()
    
    seg_config = SegmentationConfig()
    seg_config.OUTPUT_FORMAT = "PNG"
    segmenter = LeafSegmenter(seg_config)

    class_config = ClassificationConfig()
    if torch.cuda.is_available():
        class_config.DEVICE = "cuda"
    else:
        class_config.DEVICE = "cpu"

    classifier = DiseaseClassifier(class_config)
    weights_path = Path(args.weights)
    if not weights_path.exists():
        logger.error(f"Classifier weights not found at: {weights_path}")
        return

    classifier.load_state_dict(torch.load(weights_path, map_location=class_config.DEVICE))
    classifier.to(class_config.DEVICE)
    classifier.eval()

    # 2. Load Class Mappings
    mapping_file = class_config.OUTPUT_DIRECTORY / "class_mapping.json"
    if not mapping_file.exists():
        logger.error(f"Class mapping file not found at: {mapping_file}")
        return
        
    with open(mapping_file, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)
    index_to_class = {idx: name for name, idx in class_mapping.items()}

    # 3. Create analysis directory
    failed_dir = Path("analysis/failed_cases")
    if failed_dir.exists():
        shutil.rmtree(failed_dir)
    failed_dir.mkdir(parents=True, exist_ok=True)

    # 4. Read validation files
    source_path = Path(args.source)
    labels_path = Path(args.labels)
    
    if not source_path.exists():
        logger.error(f"Source directory not found: {source_path}")
        return

    img_files = sorted([
        f for f in source_path.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if args.limit:
        img_files = img_files[:args.limit]

    logger.info(f"Running pipeline on {len(img_files)} real-world validation images...")
    
    correct_count = 0
    total_count = 0
    unknown_rejections = 0
    failed_cases_logged = 0

    for img_file in tqdm(img_files, desc="Validating Pipeline"):
        try:
            # Load original image
            pil_img = Image.open(img_file).convert("RGB")
            w, h = pil_img.size

            # Load ground-truth labels
            label_file = labels_path / f"{img_file.stem}.txt"
            gt_boxes = parse_ground_truth(label_file, w, h)

            if not gt_boxes:
                continue

            # Run YOLO leaf detector
            detections = detector.detect(img_file)
            if not detections:
                continue

            # Process each detection
            for det_idx, det in enumerate(detections):
                det_box = (det.x1, det.y1, det.x2, det.y2)
                
                # Find matching ground truth to get true label
                best_gt = None
                max_iou = 0.0
                for gt in gt_boxes:
                    iou = calculate_iou(det_box, gt["box"])
                    if iou > max_iou:
                        max_iou = iou
                        best_gt = gt

                if not best_gt or max_iou < 0.3 or best_gt["mapped_class"] == "Unknown":
                    continue

                true_class = best_gt["mapped_class"]
                total_count += 1

                # Run SAM2 Segmentation (background removal)
                segmented_crop = segmenter.segment_leaf(pil_img, det_box)
                
                # Save temp crop for classifier prediction
                temp_crop_path = Path(f"temp_val_crop_{det_idx}.png")
                segmented_crop.save(temp_crop_path, "PNG")

                # Run classification
                pred_class, confidence, _ = predict_disease(
                    classifier, temp_crop_path, index_to_class, class_config.DEVICE, ood_threshold=args.threshold
                )
                
                # Clean up temp crop
                if temp_crop_path.exists():
                    temp_crop_path.unlink()

                # Evaluate
                if pred_class == "Unknown Disease":
                    unknown_rejections += 1
                    
                if pred_class == true_class:
                    correct_count += 1
                else:
                    # Logging failed cases
                    failed_cases_logged += 1
                    # Draw error box on original image crop
                    error_img = draw_error_overlay(pil_img, det_box, true_class, pred_class, confidence)
                    error_img_file = failed_dir / f"fail_{img_file.stem}_crop_{det_idx + 1}.jpg"
                    error_img.save(error_img_file, "JPEG")

        except Exception as e:
            logger.error(f"Error validating image {img_file.name}: {e}")

    # 5. Output summary metrics
    accuracy = (correct_count / total_count * 100.0) if total_count > 0 else 0.0
    
    print("\n" + "=" * 60)
    print("PHASE 4.6: REAL-WORLD PIPELINE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total Leaf Detections Evaluated : {total_count}")
    print(f"Correct Classifications         : {correct_count}")
    print(f"OOD 'Unknown' Rejections        : {unknown_rejections}")
    print(f"Wrongly Classified (Log/Fail)   : {failed_cases_logged}")
    print(f"Pipeline Generalization Accuracy: {accuracy:.2f}%")
    print("-" * 60)
    print(f"Failed cases saved to directory : {failed_dir.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
