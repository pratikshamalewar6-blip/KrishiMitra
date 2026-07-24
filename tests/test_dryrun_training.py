"""
KrishiMitra - Dry-Run Training Test

Verifies the end-to-end two-stage fine-tuning execution using a 2-batch dry run on CPU.

Author:
    Antigravity AI
"""

from __future__ import annotations

import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from classification.config import ClassificationConfig
from classification.model import DiseaseClassifier
from data.merged_dataset import MergedDiseaseDataset
from data.transforms import get_train_transforms, get_validation_transforms
from classification.losses import get_loss_criterion


class TestTrainingDryRun(unittest.TestCase):
    def test_dry_run(self):
        print("\nStarting Training Dry Run Test...")
        
        # 1. Setup config
        config = ClassificationConfig()
        config.DEVICE = "cpu"
        config.BATCH_SIZE = 4  # Very small batch size for dry run
        
        # 2. Setup datasets
        train_dataset = MergedDiseaseDataset(
            split="train",
            transform=get_train_transforms(),
            mix_ratio=0.15
        )
        val_dataset = MergedDiseaseDataset(
            split="val",
            transform=get_validation_transforms(),
            mix_ratio=0.15
        )

        train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

        # 3. Create loss criterion
        class_counts = train_dataset.get_class_counts()
        criterion = get_loss_criterion(
            loss_name="WeightedCrossEntropy",
            class_counts=class_counts,
            num_classes=config.NUM_CLASSES,
            device=config.DEVICE
        )

        # 4. Load model
        model = DiseaseClassifier(config)
        model.to(config.DEVICE)

        # ------------------------------------------------------
        # Dry-run Stage 1: Head training (Backbone Frozen)
        # ------------------------------------------------------
        print("Verifying Stage 1 gradient flow...")
        for param in model.base_model.features.parameters():
            param.requires_grad = False
        for param in model.base_model.classifier.parameters():
            param.requires_grad = True

        optimizer = optim.AdamW(model.base_model.classifier.parameters(), lr=1e-3)
        
        # Process 2 batches
        model.train()
        batch_idx = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Verify loss is scalar and not NaN
            self.assertFalse(torch.isnan(loss))
            print(f"  Stage 1 Batch {batch_idx + 1} Loss: {loss.item():.4f}")
            
            batch_idx += 1
            if batch_idx >= 2:
                break

        # Verify backbone gradients are None (frozen)
        for param in list(model.base_model.features.parameters())[:10]:
            self.assertTrue(param.grad is None)

        # Verify classifier gradients are computed (trainable)
        for param in model.base_model.classifier.parameters():
            self.assertTrue(param.grad is not None)

        # ------------------------------------------------------
        # Dry-run Stage 2: Fine-Tuning (Unfreeze last 3 blocks)
        # ------------------------------------------------------
        print("Verifying Stage 2 gradient flow...")
        for i in [6, 7, 8]:
            for param in model.base_model.features[i].parameters():
                param.requires_grad = True

        backbone_params = []
        for i in [6, 7, 8]:
            backbone_params.extend(list(model.base_model.features[i].parameters()))
        head_params = list(model.base_model.classifier.parameters())

        optimizer_stage2 = optim.AdamW([
            {"params": backbone_params, "lr": 1e-5},
            {"params": head_params, "lr": 1e-4}
        ])

        # Process 2 batches
        model.train()
        batch_idx = 0
        for images, labels in train_loader:
            optimizer_stage2.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_stage2.step()
            
            self.assertFalse(torch.isnan(loss))
            print(f"  Stage 2 Batch {batch_idx + 1} Loss: {loss.item():.4f}")
            
            batch_idx += 1
            if batch_idx >= 2:
                break

        # Verify unfrozen backbone gradients are computed
        for param in model.base_model.features[8].parameters():
            self.assertTrue(param.grad is not None)

        # ------------------------------------------------------
        # Dry-run Validation
        # ------------------------------------------------------
        print("Verifying validation pass...")
        model.eval()
        batch_idx = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                self.assertFalse(torch.isnan(loss))
                print(f"  Validation Batch {batch_idx + 1} Loss: {loss.item():.4f}")
                
                batch_idx += 1
                if batch_idx >= 2:
                    break

        print("Training dry run completed successfully!")


if __name__ == "__main__":
    unittest.main()
