"""
KrishiMitra - Real-World Classifier Training Pipeline

Fine-tunes EfficientNet-B0 on a merged dataset of PlantVillage and PlantDoc crops.
Uses a two-stage training strategy:
1. Freeze backbone features, train custom classification head (high LR).
2. Unfreeze the last 3 blocks of the backbone, fine-tune end-to-end (low LR).

Author:
    Antigravity AI
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.logger import LoggerManager
from common.file_utils import FileUtils
from classification.config import ClassificationConfig
from classification.model import DiseaseClassifier
from data.merged_dataset import MergedDiseaseDataset
from data.transforms import get_train_transforms, get_validation_transforms
from classification.losses import get_loss_criterion

logger = LoggerManager.get_logger("TrainRealWorld")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    is_training: bool = True
) -> tuple[float, float]:
    """Runs a single epoch of training or validation."""
    if is_training:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    desc = "Training Batches" if is_training else "Validation Batches"
    
    # Disable tqdm leaves to keep logs cleaner
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images, labels = images.to(device), labels.to(device)

        if is_training:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B0 on Merged Real-World Dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of fine-tuning epochs (Stage 2)")
    parser.add_argument("--stage1-epochs", type=int, default=5, help="Number of head training epochs (Stage 1)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Stage 1 head learning rate")
    parser.add_argument("--backbone-lr", type=float, default=0.00001, help="Stage 2 backbone learning rate")
    parser.add_argument("--mix-ratio", type=float, default=0.15, help="Oversampling ratio for PlantDoc crops")
    parser.add_argument("--pretrained-path", type=str, default="saved_models/efficientnet_b0_disease.pt", help="Path to pre-trained Stage 1 PlantVillage checkpoint")
    parser.add_argument("--loss", type=str, default="WeightedCrossEntropy", choices=["WeightedCrossEntropy", "FocalLoss"], help="Loss criterion")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu', 'cuda', etc.)")
    args = parser.parse_args()

    logger.info("Initializing Real-World Fine-Tuning Pipeline...")

    # Load configuration
    config = ClassificationConfig()
    
    # Override settings
    if args.device:
        config.DEVICE = args.device
    elif torch.cuda.is_available():
        config.DEVICE = "cuda"
    else:
        config.DEVICE = "cpu"

    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    
    # Model path for realworld checkpoint
    realworld_model_path = Path("saved_models/efficientnet_b0_realworld.pt")
    realworld_model_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Merged Datasets
    logger.info(f"Loading merged train dataset (mix ratio: {args.mix_ratio})...")
    train_dataset = MergedDiseaseDataset(
        split="train",
        transform=get_train_transforms(),
        mix_ratio=args.mix_ratio
    )
    val_dataset = MergedDiseaseDataset(
        split="val",
        transform=get_validation_transforms(),
        mix_ratio=args.mix_ratio
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # 2. Get class counts and construct loss criterion
    class_counts = train_dataset.get_class_counts()
    criterion = get_loss_criterion(
        loss_name=args.loss,
        class_counts=class_counts,
        num_classes=config.NUM_CLASSES,
        device=config.DEVICE
    )
    logger.info(f"Loss criterion initialized: {args.loss}")

    # 3. Build model and move to device
    model = DiseaseClassifier(config)
    
    # Load Stage 1 PlantVillage pre-trained weights if available
    pretrained_file = Path(args.pretrained_path)
    if pretrained_file.exists():
        logger.info(f"Loading pre-trained PlantVillage weights from: {pretrained_file}")
        state_dict = torch.load(pretrained_file, map_location=config.DEVICE)
        model.load_state_dict(state_dict)
    else:
        logger.info(f"Pre-trained weights file '{pretrained_file}' not found. Initializing from ImageNet backbone.")

    model.to(config.DEVICE)

    # ==========================================================
    # STAGE 1: Train Head Only (Backbone Frozen)
    # ==========================================================
    logger.info("==================================================")
    logger.info(f"STAGE 1: Training Head Only for {args.stage1_epochs} Epochs")
    logger.info("==================================================")

    # Freeze entire backbone
    for param in model.base_model.features.parameters():
        param.requires_grad = False
    
    # Ensure classification head is trainable
    for param in model.base_model.classifier.parameters():
        param.requires_grad = True

    optimizer_stage1 = optim.AdamW(
        model.base_model.classifier.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY
    )

    for epoch in range(1, args.stage1_epochs + 1):
        start_time = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer_stage1, config.DEVICE, is_training=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, config.DEVICE, is_training=False)
        epoch_time = time.time() - start_time
        
        logger.info(
            f"Stage 1 | Epoch {epoch}/{args.stage1_epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

    # ==========================================================
    # STAGE 2: Fine-Tuning (Unfreeze Last 3 Backbone Blocks)
    # ==========================================================
    logger.info("==================================================")
    logger.info(f"STAGE 2: Fine-Tuning last 3 blocks for {args.epochs} Epochs")
    logger.info("==================================================")

    # Unfreeze the last 3 blocks (layers 6, 7, 8 in features)
    for i in [6, 7, 8]:
        for param in model.base_model.features[i].parameters():
            param.requires_grad = True

    # Assemble parameter groups with differential learning rates
    backbone_params = []
    for i in [6, 7, 8]:
        backbone_params.extend(list(model.base_model.features[i].parameters()))
    head_params = list(model.base_model.classifier.parameters())

    optimizer_stage2 = optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.lr * 0.1}  # Lower head LR during fine-tuning
    ], weight_decay=config.WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=args.epochs)

    best_val_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer_stage2, config.DEVICE, is_training=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, config.DEVICE, is_training=False)
        scheduler.step()
        epoch_time = time.time() - start_time

        logger.info(
            f"Stage 2 | Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # Save checkpoint on validation accuracy improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), realworld_model_path)
            logger.info(f"==> New best validation accuracy: {val_acc*100:.2f}%. Saved model weights.")
        else:
            epochs_no_improve += 1

        if config.EARLY_STOPPING_ENABLED and epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break

    logger.info(f"Fine-tuning complete. Best Validation Accuracy: {best_val_acc*100:.2f}%")
    logger.info(f"Trained model saved to: {realworld_model_path}")


if __name__ == "__main__":
    main()
