# ==========================================================
# 🌿 KRISHIMITRA: Notebook 3 - Real-World Field Fine-Tuning
# ==========================================================

# 1. GPU Check & Dependencies Installation
# Run these in your Google Colab code cell:
# !nvidia-smi
# !pip install -q torch torchvision timm albumentations scikit-learn tqdm pyyaml pillow

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from pathlib import Path

# Import our custom Zero-Copy Merged Dataset
from data.merged_dataset import MergedDiseaseDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('🚀 Using Device:', device)

# 2. Heavy Field-Noise Albumentations for Real Farm Robustness
train_aug = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, border_mode=0, p=0.7),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.6),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7)),
        A.MotionBlur(blur_limit=(3, 7)),
    ], p=0.4),
    A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16), hole_width_range=(8, 16), p=0.4),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_aug = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Albumentations Dataset Wrapper
class AlbumentationsDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        img_np = np.array(img.convert("RGB"))
        augmented = self.transform(image=img_np)["image"]
        return augmented, label

print("\n🚀 Loading Hybrid Datasets (PlantVillage + PlantDoc via Zero-Copy)...")

# 3. Load Zero-Copy Merged Datasets with 15% PlantDoc Real-World Mix Ratio
train_dataset_raw = MergedDiseaseDataset(split="train", transform=None, mix_ratio=0.15)
val_dataset_raw = MergedDiseaseDataset(split="val", transform=None, mix_ratio=0.15)

num_classes = len(train_dataset_raw.class_to_index)

# Save Master Class Mapping in saved_models and outputs/classification
save_dir = Path("saved_models")
save_dir.mkdir(parents=True, exist_ok=True)
outputs_dir = Path("outputs/classification")
outputs_dir.mkdir(parents=True, exist_ok=True)

for m_path in [save_dir / "class_mapping.json", outputs_dir / "class_mapping.json"]:
    with open(m_path, "w", encoding="utf-8") as f:
        json.dump(train_dataset_raw.class_to_index, f, indent=4)
    print(f"📖 Saved Class Mapping to: {m_path}")

# Wrap with Albumentations
train_dataset = AlbumentationsDataset(train_dataset_raw, train_aug)
val_dataset = AlbumentationsDataset(val_dataset_raw, val_aug)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

print(f"✅ Training Samples Loaded: {len(train_dataset)}")
print(f"✅ Validation Samples Loaded: {len(val_dataset)}")
print(f"✅ Total Output Classes: {num_classes}")

# 4. EfficientNet-B0 Architecture Setup
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()

# --- STAGE 1: Head Training (5 Epochs) ---
print("\n==========================================")
print("STAGE 1: Head Training (5 Epochs, LR=1e-3)")
print("==========================================")
for param in model.features.parameters():
    param.requires_grad = False

optimizer_s1 = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(1, 6):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in tqdm(train_loader, desc=f"Stage 1 Epoch {epoch}/5"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer_s1.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer_s1.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outs.max(1)
        correct += preds.eq(lbls).sum().item()
        total += lbls.size(0)
    
    # Stage 1 Validation Evaluation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            loss = criterion(outs, lbls)
            val_loss += loss.item() * imgs.size(0)
            _, preds = outs.max(1)
            val_correct += preds.eq(lbls).sum().item()
            val_total += lbls.size(0)
            
    print(f"Stage 1 | Epoch {epoch}/5 | Train Loss: {running_loss/total:.4f} | Train Acc: {(correct/total)*100:.2f}% | Val Loss: {val_loss/val_total:.4f} | Val Acc: {(val_correct/val_total)*100:.2f}%")

# --- STAGE 2: Real-World Backbone Fine-Tuning (20 Epochs) ---
print("\n==========================================")
print("STAGE 2: Field Fine-Tuning Backbone (20 Epochs, LR=1e-5)")
print("==========================================")
for param in model.features[-3:].parameters():
    param.requires_grad = True

optimizer_s2 = optim.AdamW([
    {"params": model.features[-3:].parameters(), "lr": 1e-5},
    {"params": model.classifier.parameters(), "lr": 1e-4}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_s2, T_max=20)

best_acc = 0.0
model_save_path = save_dir / "efficientnet_b0_disease.pt"
realworld_save_path = save_dir / "efficientnet_b0_realworld.pt"

for epoch in range(1, 21):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for imgs, lbls in tqdm(train_loader, desc=f"Stage 2 Epoch {epoch}/20"):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer_s2.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer_s2.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outs.max(1)
        correct += preds.eq(lbls).sum().item()
        total += lbls.size(0)
    scheduler.step()

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            loss = criterion(outs, lbls)
            val_loss += loss.item() * imgs.size(0)
            _, preds = outs.max(1)
            val_correct += preds.eq(lbls).sum().item()
            val_total += lbls.size(0)
            
    val_acc = val_correct / val_total
    print(f"Stage 2 | Epoch {epoch:02d}/20 | Train Acc: {(correct/total)*100:.2f}% | Val Loss: {val_loss/val_total:.4f} | Val Acc: {val_acc*100:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        torch.save(model.state_dict(), realworld_save_path)
        print(f"  ==> 🌟 Saved New Best Model Weights ({val_acc*100:.2f}%) to: {model_save_path}")

print(f"\n🎉 Real-World Field Fine-Tuning Complete! Best Validation Accuracy: {best_acc*100:.2f}%")
print(f"💾 Model weights successfully saved at:\n  - {model_save_path}\n  - {realworld_save_path}")
