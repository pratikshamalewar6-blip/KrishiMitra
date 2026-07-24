"""
KrishiMitra - Crop Disease Classifier Training Pipeline

Trains EfficientNet-B0 on crop disease datasets (PlantVillage).

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
from tqdm import tqdm

from common.logger import LoggerManager
from common.file_utils import FileUtils
from classification.config import ClassificationConfig
from classification.model import DiseaseClassifier
from classification.dataset import DataLoaderFactory

# Optional TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


def train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """
    Runs a single training epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training Batches", leave=False):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, float]:
    """
    Runs a single validation epoch.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation Batches", leave=False):
        images, labels = images.to(device), labels.to(device)

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
    parser = argparse.ArgumentParser(description="Train EfficientNet-B0 Disease Classifier")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu', 'cuda', etc.)")
    parser.add_argument("--dataset", type=str, default="plantvillage", help="Dataset name ('plantvillage', etc.)")
    args = parser.parse_args()

    logger = LoggerManager.get_logger("TrainClassifier")
    logger.info("Initializing Classification Training Pipeline...")

    # Load configurations
    config = ClassificationConfig()
    
    # Overwrite configuration with CLI args if specified
    if args.epochs is not None:
        config.EPOCHS = args.epochs
    if args.batch_size is not None:  # CLI argparse mapping translates hyphens to underscores
        config.BATCH_SIZE = args.batch_size
    if args.lr is not None:
        config.LEARNING_RATE = args.lr
    if args.device is not None:
        config.DEVICE = args.device

    # Ensure output directories exist
    FileUtils.ensure_directory(config.OUTPUT_DIRECTORY)
    config.MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Target Device  : {config.DEVICE}")
    logger.info(f"Batch Size     : {config.BATCH_SIZE}")
    logger.info(f"Learning Rate  : {config.LEARNING_RATE}")
    logger.info(f"Total Epochs   : {config.EPOCHS}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.RANDOM_SEED)

    # 1. Initialize DataLoaders
    logger.info(f"Loading '{args.dataset}' splits...")
    loader_factory = DataLoaderFactory()
    # Override batch size in loader factory
    loader_factory.batch_size = config.BATCH_SIZE
    
    try:
        train_loader, val_loader, _ = loader_factory.create_dataloaders(args.dataset)
        # Save class mapping JSON
        class_mapping = train_loader.dataset.get_class_mapping()
        mapping_file = config.OUTPUT_DIRECTORY / "class_mapping.json"
        import json
        with open(mapping_file, "w", encoding="utf-8") as f:
            json.dump(class_mapping, f, indent=4)
        logger.info(f"Saved class mapping JSON to: {mapping_file}")
    except Exception as e:
        logger.error(f"Failed to load dataset splits: {e}")
        logger.info("Please make sure split CSV files exist under outputs/splits/.")
        raise e

    # 2. Build model and move to device
    model = DiseaseClassifier(config)
    model.to(config.DEVICE)

    # 3. Criterion, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    
    if config.OPTIMIZER_NAME == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    if config.SCHEDULER_NAME == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    else:
        scheduler = None

    # 4. Setup TensorBoard SummaryWriter
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer_path = config.TENSORBOARD_DIRECTORY / f"efficientnet_b0_{args.dataset}_{int(time.time())}"
        writer = SummaryWriter(log_dir=str(writer_path))
        logger.info(f"TensorBoard logging enabled. Run: tensorboard --logdir={config.TENSORBOARD_DIRECTORY}")

    # 5. Training Loop
    best_val_acc = 0.0
    epochs_no_improve = 0

    logger.info("Starting training loop...")
    for epoch in range(1, config.EPOCHS + 1):
        epoch_start_time = time.time()

        # Train and validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config.DEVICE)

        if scheduler:
            scheduler.step()

        # Log metrics
        epoch_time = time.time() - epoch_start_time
        logger.info(
            f"Epoch {epoch:02d}/{config.EPOCHS:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )

        # TensorBoard write
        if writer:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Val", val_acc, epoch)
            if scheduler:
                writer.add_scalar("LearningRate", scheduler.get_last_lr()[0], epoch)

        # Checkpoint: Save best model weights
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.MODEL_FILE)
            logger.info(f"==> New best validation accuracy: {val_acc*100:.2f}%. Saved model weights.")
        else:
            epochs_no_improve += 1

        # Early Stopping
        if config.EARLY_STOPPING_ENABLED and epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered after {epoch} epochs of no improvement.")
            break

    # Save final last checkpoint
    last_checkpoint_path = config.MODEL_FILE.with_name("efficientnet_b0_disease_last.pt")
    torch.save(model.state_dict(), last_checkpoint_path)
    logger.info(f"Saved last model weights checkpoint to: {last_checkpoint_path}")

    if writer:
        writer.close()

    logger.info("Disease Classifier training completed successfully!")


if __name__ == "__main__":
    main()
