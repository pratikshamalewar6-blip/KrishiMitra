# Implementation Plan - Phase 4: EfficientNet-B0 Classification

This plan covers the design and implementation of the **EfficientNet-B0 Crop Disease Classification** module. The goal of this module is to categorize a segmented leaf image into one of 38 categories representing healthy plants or specific plant diseases.

## User Review Required

> [!IMPORTANT]
> **Dataset Selection:** By default, training will run on the processed **PlantVillage** dataset, which consists of 38 classes. Since training deep networks on CPU is slow, we will default local training to a quick verification run of **1 epoch**, and provide arguments for full GPU training.
>
> **Data Augmentation:** Crop leaf images will be resized to **224x224**, normalized, and augmented using standard operations (flips, random rotation, brightness/contrast adjustments) using `albumentations` or `torchvision.transforms` to improve generalization.

## Proposed Changes

We will implement the classification files in the `classification/` component and add a unit test script.

---

### 1. Configuration Component
#### [NEW] [config.py](file:///d:/KrishiMitra2/ai_models/disease_detection/classification/config.py)
* Class `ClassificationConfig` loaded from `configs/model.yaml` and `configs/training.yaml`.
* Defines learning rate, epochs, batch size, architecture, input size (224x224), early stopping patience, and checkpoint directories.

---

### 2. Dataset Loader Component
#### [NEW] [dataset.py](file:///d:/KrishiMitra2/ai_models/disease_detection/classification/dataset.py)
* Custom PyTorch `Dataset` wrapper for classification.
* Prepares train, val, and test splits.
* Incorporates Albumentations/Torchvision pipelines for training augmentations and validation preprocessing.

---

### 3. Model Architecture Component
#### [NEW] [model.py](file:///d:/KrishiMitra2/ai_models/disease_detection/classification/model.py)
* Class `DiseaseClassifier` inheriting from `torch.nn.Module`.
* Wraps a pretrained `efficientnet_b0` from torchvision or timm.
* Replaces the final classifier linear head with a mapping to the 38 crop-disease class outputs.

---

### 4. Training Loop Component
#### [NEW] [train_classifier.py](file:///d:/KrishiMitra2/ai_models/disease_detection/classification/train_classifier.py)
* Script to run model training.
* Implements loss calculation (CrossEntropyLoss) and optimizer (AdamW).
* Tracks training and validation metrics per epoch.
* Implements checkpoints and early stopping based on validation accuracy/loss.
* Saves weights to `saved_models/efficientnet_b0_disease.pt`.

---

### 5. Evaluation & Inference Component
#### [NEW] [evaluate_classifier.py](file:///d:/KrishiMitra2/ai_models/disease_detection/classification/evaluate_classifier.py)
* Loads `saved_models/efficientnet_b0_disease.pt` and evaluates it on the test split.
* Generates classification reports (Precision, Recall, F1) and saves them in `outputs/classification/`.
* Computes and saves a confusion matrix plot.

#### [NEW] [predict_classifier.py](file:///d:/KrishiMitra2/ai_models/disease_detection/classification/predict_classifier.py)
* CLI prediction script to classify diseases in a cropped leaf image.

---

### 6. Verification Component
#### [NEW] [test_classifier_model.py](file:///d:/KrishiMitra2/ai_models/disease_detection/tests/test_classifier_model.py)
* Unit test file to verify classifier architecture, output logits dimensions, and forward pass on mock inputs.

---

## Verification Plan

### Automated Tests
1. Run classification model unit test:
   ```powershell
   $env:PYTHONPATH="."
   .\venv\Scripts\python.exe tests/test_classifier_model.py
   ```
2. Verify 1-epoch training pipeline execution:
   ```powershell
   $env:PYTHONPATH="."
   .\venv\Scripts\python.exe classification/train_classifier.py --epochs 1 --batch-size 8
   ```
3. Run prediction on a single crop:
   ```powershell
   $env:PYTHONPATH="."
   .\venv\Scripts\python.exe classification/predict_classifier.py --image outputs/segmentations/val_00000_mock_segmented.png
   ```
