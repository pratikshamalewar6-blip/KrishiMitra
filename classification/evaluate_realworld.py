"""
KrishiMitra - Real-World Classifier Evaluation & Comparison

Compares the old model (trained on PlantVillage) against the improved real-world model
(trained on the merged dataset) across both PlantVillage and PlantDoc test sets.
Computes Top-1/Top-5 accuracy, inference latency, and size metrics.

Author:
    Antigravity AI
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from common.logger import LoggerManager
from classification.config import ClassificationConfig
from classification.model import DiseaseClassifier
from data.merged_dataset import MergedDiseaseDataset
from data.transforms import get_validation_transforms

logger = LoggerManager.get_logger("EvaluateRealWorld")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> tuple[float, float, dict]:
    """Evaluates a model and returns Top-1 accuracy, Top-5 accuracy, and the classification report dict."""
    model.eval()
    
    all_targets = []
    all_preds = []
    
    top1_correct = 0
    top5_correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        
        # Calculate Top-1 and Top-5
        _, top5_indices = outputs.topk(min(5, outputs.size(1)), dim=1, largest=True, sorted=True)
        
        for i, target in enumerate(labels):
            total += 1
            if target == top5_indices[i, 0]:
                top1_correct += 1
            if target in top5_indices[i]:
                top5_correct += 1
                
            all_targets.append(int(target.item()))
            all_preds.append(int(top5_indices[i, 0].item()))

    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    
    # Generate classification report dict
    report = classification_report(
        all_targets,
        all_preds,
        output_dict=True,
        zero_division=0
    )
    
    return top1_acc, top5_acc, report


def measure_inference_latency(model: nn.Module, device: str, input_size: int = 224, num_runs: int = 100) -> float:
    """Measures average forward pass latency in milliseconds."""
    model.eval()
    mock_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(mock_input)
            
    # Measure
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(mock_input)
            
    latency_ms = ((time.time() - start_time) / num_runs) * 1000.0
    return latency_ms


def main() -> None:
    config = ClassificationConfig()
    
    if torch.cuda.is_available():
        config.DEVICE = "cuda"
    else:
        config.DEVICE = "cpu"

    logger.info(f"Evaluating models on device: {config.DEVICE}")

    # 1. Load Test Datasets
    transform = get_validation_transforms()
    
    logger.info("Loading PlantVillage test split...")
    pv_test_dataset = MergedDiseaseDataset(split="test", transform=transform, mix_ratio=0.0)
    pv_loader = DataLoader(pv_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    
    logger.info("Loading PlantDoc test split...")
    pd_test_dataset = MergedDiseaseDataset(split="test", transform=transform, mix_ratio=1.0)
    pd_loader = DataLoader(pd_test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # 2. Setup models
    old_model_path = Path("saved_models/efficientnet_b0_disease.pt")
    new_model_path = Path("saved_models/efficientnet_b0_realworld.pt")

    results = {}

    model_paths = {
        "Old Model (PlantVillage only)": old_model_path,
        "Improved Model (Real-World)": new_model_path
    }

    for model_name, path in model_paths.items():
        if not path.exists():
            logger.warning(f"{model_name} weights file not found at: {path}. Skipping evaluation for this model.")
            continue

        logger.info(f"--- Evaluating {model_name} ---")
        
        # Load weights
        model = DiseaseClassifier(config)
        model.load_state_dict(torch.load(path, map_location=config.DEVICE))
        model.to(config.DEVICE)
        
        # Benchmarks
        pv_top1, pv_top5, pv_report = evaluate_model(model, pv_loader, config.DEVICE)
        pd_top1, pd_top5, pd_report = evaluate_model(model, pd_loader, config.DEVICE)
        latency = measure_inference_latency(model, config.DEVICE)
        model_size_mb = path.stat().st_size / (1024 * 1024)

        results[model_name] = {
            "model_size_mb": model_size_mb,
            "inference_latency_ms": latency,
            "plantvillage_top1": pv_top1,
            "plantvillage_top5": pv_top5,
            "plantdoc_top1": pd_top1,
            "plantdoc_top5": pd_top5
        }

    if len(results) < 2:
        logger.warning("Need both old and improved models to generate a comparison report.")
        return

    # 3. Print Comparison Report
    print("\n" + "=" * 80)
    print("COMPARATIVE EVALUATION REPORT: OLD VS IMPROVED")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Old Model':<18} | {'Improved Model':<18} | {'Diff':<8}")
    print("-" * 80)
    
    metrics = [
        ("Model Size (MB)", "model_size_mb", "{:.2f} MB", False),
        ("Inference Latency (ms)", "inference_latency_ms", "{:.2f} ms", False),
        ("PlantVillage Top-1 Acc", "plantvillage_top1", "{:.2f}%", True),
        ("PlantVillage Top-5 Acc", "plantvillage_top5", "{:.2f}%", True),
        ("PlantDoc Top-1 Acc", "plantdoc_top1", "{:.2f}%", True),
        ("PlantDoc Top-5 Acc", "plantdoc_top5", "{:.2f}%", True),
    ]

    report_data = []

    for label, key, fmt, is_percentage in metrics:
        old_val = results["Old Model (PlantVillage only)"][key]
        new_val = results["Improved Model (Real-World)"][key]
        
        diff = (new_val - old_val)
        if is_percentage:
            old_str = fmt.format(old_val * 100.0)
            new_str = fmt.format(new_val * 100.0)
            diff_str = f"{diff * 100.0:+.2f}%"
        else:
            old_str = fmt.format(old_val)
            new_str = fmt.format(new_val)
            diff_str = f"{diff:+.2f}"

        print(f"{label:<30} | {old_str:<18} | {new_str:<18} | {diff_str:<8}")
        report_data.append({
            "metric": label,
            "old_value": old_str,
            "new_value": new_str,
            "difference": diff_str
        })

    print("=" * 80 + "\n")

    # Save reports
    output_dir = Path("outputs/classification")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    with open(output_dir / "realworld_comparison_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    # Save Markdown report
    md_file = output_dir / "realworld_comparison_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# KrishiMitra Classifier Comparison Report\n\n")
        f.write("A comparison between the base model trained only on PlantVillage and the fine-tuned real-world model.\n\n")
        f.write("| Metric | Old Model | Improved Model (Real-World) | Difference |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for item in report_data:
            f.write(f"| {item['metric']} | {item['old_value']} | {item['new_value']} | {item['difference']} |\n")
            
    logger.info(f"Markdown report generated at: {md_file.resolve()}")


if __name__ == "__main__":
    main()
