"""
KrishiMitra - Crop Disease Classifier Prediction CLI (with Calibration & OOD)

Classifies crop diseases on single leaf images with Top-5 scores and OOD detection.

Author:
    Antigravity AI
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

from common.logger import LoggerManager
from data.transforms import get_test_transforms
from classification.config import ClassificationConfig
from classification.model import DiseaseClassifier

logger = LoggerManager.get_logger("PredictClassifier")


def predict_disease(
    model: torch.nn.Module,
    image_path: Path,
    index_to_class: dict[int, str],
    device: str,
    ood_threshold: float = 0.60
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Runs prediction on a single image.
    
    Returns
    -------
    predicted_class : str
        The predicted disease class (or 'Unknown Disease' if confidence < ood_threshold)
    confidence : float
        The top-1 confidence value
    top5_predictions : list[tuple[str, float]]
        List of (class_name, probability) for the top-5 classes
    """
    # 1. Load and transform image
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image {image_path}: {e}")
        raise e

    # Retrieve test transform (Resize 224x224, Normalization)
    transform = get_test_transforms()
    img_tensor = transform(img)
    
    # Add batch dimension: (1, 3, 224, 224)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # 2. Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1).squeeze(0)

    # 3. Retrieve Top-5
    topk_probs, topk_indices = torch.topk(probabilities, k=min(5, len(probabilities)))
    
    top5_predictions = []
    for prob, idx in zip(topk_probs, topk_indices):
        prob_val = float(prob.item())
        class_name = index_to_class.get(int(idx.item()), f"Class_{idx.item()}")
        top5_predictions.append((class_name, prob_val))

    top1_class, top1_prob = top5_predictions[0]

    # 4. Out-of-Distribution (OOD) check
    if top1_prob < ood_threshold:
        logger.info(f"Top-1 confidence ({top1_prob*100:.2f}%) is below OOD threshold ({ood_threshold*100}%).")
        predicted_class = "Unknown Disease"
    else:
        predicted_class = top1_class

    return predicted_class, top1_prob, top5_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify Crop Disease on a Leaf Crop with Top-5 and OOD Rejection")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input cropped leaf image file",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to trained model weights checkpoint",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Out-of-Distribution (OOD) confidence threshold (default: 0.60)",
    )
    args = parser.parse_args()

    config = ClassificationConfig()
    
    # Override weights path if specified
    weights_path = Path(args.weights) if args.weights else config.MODEL_FILE

    if not weights_path.exists():
        # Check if realworld weights exist and fallback to it if default is missing
        rw_path = Path("saved_models/efficientnet_b0_realworld.pt")
        if rw_path.exists() and not args.weights:
            weights_path = rw_path
        else:
            logger.error(f"Weights file not found at: {weights_path}")
            logger.info("Please train the model first or specify the correct weights file.")
            return

    # 1. Load class mappings
    mapping_file = config.OUTPUT_DIRECTORY / "class_mapping.json"
    index_to_class = {}
    
    if mapping_file.exists():
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                class_mapping = json.load(f)
            index_to_class = {idx: name for name, idx in class_mapping.items()}
            logger.info(f"Loaded class mapping for {len(index_to_class)} classes.")
        except Exception as e:
            logger.warning(f"Failed to load class mapping JSON: {e}")
    else:
        logger.warning(f"Class mapping file '{mapping_file}' not found. Defaulting class names to indexes.")

    # 2. Load model and weights
    model = DiseaseClassifier(config)
    logger.info(f"Loading trained weights from: {weights_path}")
    
    try:
        model.load_state_dict(torch.load(weights_path, map_location=config.DEVICE))
        model.to(config.DEVICE)
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        return

    # 3. Predict
    img_file = Path(args.image)
    if not img_file.exists():
        logger.error(f"Input image not found: {img_file}")
        return

    logger.info(f"Predicting crop disease for: {img_file}")
    try:
        class_name, confidence, top5 = predict_disease(
            model, img_file, index_to_class, config.DEVICE, ood_threshold=args.threshold
        )
        
        print("\n" + "=" * 60)
        print("Prediction Result:")
        print(f"Disease Diagnosis : {class_name}")
        print(f"Confidence        : {confidence * 100:.2f}%")
        print("-" * 60)
        print("Top 5 Predictions:")
        for idx, (cls, prob) in enumerate(top5):
            print(f"  {idx + 1}. {cls}: {prob * 100:.2f}%")
        print("=" * 60 + "\n")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
