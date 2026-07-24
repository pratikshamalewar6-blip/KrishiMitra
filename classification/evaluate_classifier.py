"""
KrishiMitra - Crop Disease Classifier Evaluation Pipeline

Evaluates trained EfficientNet-B0 on the test split and saves evaluation reports.

Author:
    Antigravity AI
"""

from __future__ import annotations

import json
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm

from common.logger import LoggerManager
from common.file_utils import FileUtils
from classification.config import ClassificationConfig
from classification.model import DiseaseClassifier
from classification.dataset import DataLoaderFactory

# Machine Learning and Visualization
try:
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AND_VIZ_AVAILABLE = True
except ImportError:
    SKLEARN_AND_VIZ_AVAILABLE = False


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float, list[int], list[int]]:
    """
    Run prediction on test loader and return loss, accuracy, targets, and predictions.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_targets = []
    all_predictions = []

    for images, labels in tqdm(loader, desc="Evaluation Batches", leave=False):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_targets.extend(labels.cpu().tolist())
        all_predictions.extend(predicted.cpu().tolist())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_targets, all_predictions


def main() -> None:
    logger = LoggerManager.get_logger("EvaluateClassifier")
    logger.info("Initializing Classifier Evaluation Pipeline...")

    config = ClassificationConfig()
    
    # Ensure output directory exists
    FileUtils.ensure_directory(config.OUTPUT_DIRECTORY)

    logger.info(f"Target Device: {config.DEVICE}")

    # 1. Initialize Loader Factory
    loader_factory = DataLoaderFactory()
    loader_factory.batch_size = config.BATCH_SIZE
    
    logger.info("Loading dataset splits...")
    try:
        # We only need the test loader
        _, _, test_loader = loader_factory.create_dataloaders("plantvillage")
    except Exception as e:
        logger.error(f"Failed to load dataset splits: {e}")
        logger.info("Please make sure split CSV files exist under outputs/splits/.")
        raise e

    # 2. Get class mapping from test dataset
    test_dataset = test_loader.dataset
    class_names = test_dataset.get_classes()
    logger.info(f"Test dataset contains {len(test_dataset)} samples across {len(class_names)} classes.")

    # 3. Load model and move to device
    model = DiseaseClassifier(config)
    
    if not config.MODEL_FILE.exists():
        logger.error(f"Trained model weights file not found: {config.MODEL_FILE}")
        logger.info("Please train the classifier first using: python classification/train_classifier.py")
        return
        
    logger.info(f"Loading trained weights from: {config.MODEL_FILE}")
    model.load_state_dict(torch.load(config.MODEL_FILE, map_location=config.DEVICE))
    model.to(config.DEVICE)

    # 4. Run Evaluation
    criterion = nn.CrossEntropyLoss()
    logger.info("Evaluating model on test dataset...")
    test_loss, test_acc, targets, predictions = evaluate_model(model, test_loader, criterion, config.DEVICE)

    logger.info("=" * 60)
    logger.info(f"Test Evaluation Results:")
    logger.info(f"Test Loss     : {test_loss:.4f}")
    logger.info(f"Test Accuracy : {test_acc*100:.2f}%")
    logger.info("=" * 60)

    # 5. Generate and Save Reports
    report_dict = {
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "total_samples": len(targets)
    }

    if SKLEARN_AND_VIZ_AVAILABLE:
        # Build classification report
        report_text = classification_report(
            targets, 
            predictions, 
            target_names=class_names, 
            output_dict=False
        )
        print("\n--- Classification Report ---")
        print(report_text)
        
        # Save metrics as dict
        metrics_dict = classification_report(
            targets, 
            predictions, 
            target_names=class_names, 
            output_dict=True
        )
        report_dict["metrics"] = metrics_dict

        # Generate Confusion Matrix
        cm = confusion_matrix(targets, predictions)
        
        # Plot and save Confusion Matrix
        plt.figure(figsize=(18, 14))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap="Blues", 
            xticklabels=class_names, 
            yticklabels=class_names
        )
        plt.title("Confusion Matrix - Crop Disease Classification")
        plt.ylabel("True Class")
        plt.xlabel("Predicted Class")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        cm_path = config.OUTPUT_DIRECTORY / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        logger.info(f"Confusion matrix plot saved to: {cm_path}")
        
    else:
        logger.warning("Scikit-learn, Matplotlib, or Seaborn is missing. Saving basic metrics only.")

    # Save JSON report
    report_path = config.OUTPUT_DIRECTORY / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)
    logger.info(f"Evaluation JSON report saved to: {report_path}")


if __name__ == "__main__":
    main()
