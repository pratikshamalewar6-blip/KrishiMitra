"""
KrishiMitra - PlantDoc Dataset Preprocessing Pipeline

Runs YOLOv11 leaf detection and SAM2 leaf segmentation on PlantDoc detection images.
Outputs 224x224 background-removed transparent crops in classification format.

Author:
    Antigravity AI
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from common.logger import LoggerManager
from common.file_utils import FileUtils
from detection.detector import LeafDetector
from segmentation.segmenter import LeafSegmenter
from segmentation.config import SegmentationConfig

logger = LoggerManager.get_logger("PreprocessPlantDoc")

# Map of PlantDoc class ID to Class Name
PLANTDOC_CLASSES = {
    0: "Cherry leaf",
    1: "Peach leaf",
    2: "Corn leaf blight",
    3: "Apple rust leaf",
    4: "Potato leaf late blight",
    5: "Strawberry leaf",
    6: "Corn rust leaf",
    7: "Tomato leaf late blight",
    8: "Tomato mold leaf",
    9: "Potato leaf early blight",
    10: "Apple leaf",
    11: "Tomato leaf yellow virus",
    12: "Blueberry leaf",
    13: "Tomato leaf mosaic virus",
    14: "Raspberry leaf",
    15: "Tomato leaf bacterial spot",
    16: "Squash Powdery mildew leaf",
    17: "grape leaf",
    18: "Corn Gray leaf spot",
    19: "Tomato Early blight leaf",
    20: "Apple Scab Leaf",
    21: "Tomato Septoria leaf spot",
    22: "Tomato leaf",
    23: "Soyabean leaf",
    24: "Bell_pepper leaf spot",
    25: "Bell_pepper leaf",
    26: "grape leaf black rot",
    27: "Potato leaf",
    28: "Tomato two spotted spider mites leaf"
}

# Map of PlantDoc Class Name to PlantVillage Target Class
PLANTDOC_TO_PLANTVILLAGE = {
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Peach leaf": "Peach___healthy",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Potato leaf late blight": "Potato___Late_blight",
    "Strawberry leaf": "Strawberry___healthy",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Potato leaf early blight": "Potato___Early_blight",
    "Apple leaf": "Apple___healthy",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Blueberry leaf": "Blueberry___healthy",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Raspberry leaf": "Raspberry___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "grape leaf": "Grape___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "grape leaf black rot": "Grape___Black_rot",
    "Potato leaf": "Potato___healthy",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite"
}


def calculate_iou(box1: tuple[int, int, int, int], box2: tuple[int, int, int, int]) -> float:
    """Calculates Intersection over Union (IoU) of two bounding boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x1_i >= x2_i or y1_i >= y2_i:
        return 0.0

    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area_1 + area_2 - intersection_area

    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


def parse_ground_truth(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Parses YOLO format label files and returns absolute coordinates and class names."""
    gt_boxes = []
    if not label_path.exists():
        return gt_boxes

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert to absolute coords
                x1 = int((x_center - width / 2) * img_w)
                y1 = int((y_center - height / 2) * img_h)
                x2 = int((x_center + width / 2) * img_w)
                y2 = int((y_center + height / 2) * img_h)

                raw_class_name = PLANTDOC_CLASSES.get(cls_id, "Unknown")
                mapped_class_name = PLANTDOC_TO_PLANTVILLAGE.get(raw_class_name, "Unknown")

                gt_boxes.append({
                    "box": (x1, y1, x2, y2),
                    "raw_class": raw_class_name,
                    "mapped_class": mapped_class_name
                })
    except Exception as e:
        logger.warning(f"Failed to parse label file {label_path}: {e}")

    return gt_boxes


def process_split(
    detector: LeafDetector,
    segmenter: LeafSegmenter,
    split: str,
    limit: int | None,
    output_dir: Path
) -> tuple[int, int]:
    """Processes a single dataset split (train or val)."""
    raw_images_dir = Path("datasets/raw/plantdoc_detection/images") / split
    raw_labels_dir = Path("datasets/raw/plantdoc_detection/labels") / split

    if not raw_images_dir.exists():
        logger.error(f"Image directory for split '{split}' not found at: {raw_images_dir}")
        return 0, 0

    image_files = sorted([
        f for f in raw_images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if limit:
        image_files = image_files[:limit]

    logger.info(f"Processing {len(image_files)} images for split: {split}")
    processed_count = 0
    crop_count = 0

    for img_file in tqdm(image_files, desc=f"Processing {split}"):
        try:
            # 1. Load original image
            pil_img = Image.open(img_file).convert("RGB")
            w, h = pil_img.size

            # 2. Load ground-truth labels
            label_file = raw_labels_dir / f"{img_file.stem}.txt"
            gt_boxes = parse_ground_truth(label_file, w, h)

            if not gt_boxes:
                continue

            # 3. Detect leaves using LeafDetector (YOLO)
            detections = detector.detect(img_file)
            
            # Map detected boxes to classes based on IoU overlap with GT
            boxes_to_process = []
            for det in detections:
                det_box = (det.x1, det.y1, det.x2, det.y2)
                best_gt = None
                max_iou = 0.0
                for gt in gt_boxes:
                    iou = calculate_iou(det_box, gt["box"])
                    if iou > max_iou:
                        max_iou = iou
                        best_gt = gt
                
                # If overlapped with GT, assign GT's mapped class name
                if best_gt and max_iou >= 0.3:
                    boxes_to_process.append({
                        "box": det_box,
                        "mapped_class": best_gt["mapped_class"]
                    })

            # Fallback: if YOLO detected nothing, use GT boxes directly
            if not boxes_to_process:
                for gt in gt_boxes:
                    boxes_to_process.append({
                        "box": gt["box"],
                        "mapped_class": gt["mapped_class"]
                    })

            # 4. Run SAM2 Segmentation on matched/fallback boxes
            for idx, item in enumerate(boxes_to_process):
                mapped_class = item["mapped_class"]
                if mapped_class == "Unknown":
                    continue

                # Segment leaf (returns transparent RGBA)
                segmented_crop = segmenter.segment_leaf(pil_img, item["box"])
                
                # Resize to 224x224
                resized_crop = segmented_crop.resize((224, 224), Image.Resampling.LANCZOS)

                # Save crop
                target_dir = output_dir / split / mapped_class
                target_dir.mkdir(parents=True, exist_ok=True)
                
                crop_file = target_dir / f"{img_file.stem}_crop_{idx + 1}.png"
                resized_crop.save(crop_file, "PNG")
                crop_count += 1

            processed_count += 1
        except Exception as e:
            logger.error(f"Failed to process image {img_file.name}: {e}")

    return processed_count, crop_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment and Prepare PlantDoc Classification Dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of processed images per split (for testing)")
    parser.add_argument("--split", type=str, default="all", choices=["train", "val", "all"], help="Split to process")
    args = parser.parse_args()

    # 1. Ensure models exist
    logger.info("Initializing YOLO detector and SAM2 segmenter...")
    detector = LeafDetector()
    
    # Configure segmenter to output transparent PNGs
    seg_config = SegmentationConfig()
    seg_config.OUTPUT_FORMAT = "PNG"
    segmenter = LeafSegmenter(seg_config)

    # 2. Set up output paths
    output_dir = Path("datasets/processed/plantdoc_classification")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Write class mapping JSON to outputs
    mapping_dir = Path("outputs/classification")
    mapping_dir.mkdir(parents=True, exist_ok=True)
    mapping_file = mapping_dir / "plantdoc_mapping.json"
    
    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(PLANTDOC_TO_PLANTVILLAGE, f, indent=4, ensure_ascii=False)
    logger.info(f"Class mapping exported to: {mapping_file}")

    # 4. Process Splits
    splits = ["train", "val"] if args.split == "all" else [args.split]
    
    total_imgs = 0
    total_crops = 0
    for split in splits:
        logger.info(f"--- Processing split: {split} ---")
        imgs, crops = process_split(detector, segmenter, split, args.limit, output_dir)
        total_imgs += imgs
        total_crops += crops
        logger.info(f"Split '{split}' completed: Processed {imgs} images, generated {crops} leaf crops.")

    logger.info("============================================================")
    logger.info(f"PlantDoc processing finished!")
    logger.info(f"Total Source Images Processed: {total_imgs}")
    logger.info(f"Total Segmented Leaf Crops Generated: {total_crops}")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
