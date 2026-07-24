# Implementation Plan - Phase 4: EfficientNet-B0 Classification

This plan covers the design, validation, and packaging of the **EfficientNet-B0 Crop Disease Classification** module. The goal of this module is to categorize a segmented leaf image into one of 38 categories representing healthy crops or specific crop diseases from the PlantVillage dataset.

## User Review Required

> [!IMPORTANT]
> **GPU Training Requirement:** The disease classifier has 38 output classes and will be trained on the **PlantVillage** dataset (~38,000 training images). Training this deep network on a local CPU is extremely slow. We will use **Google Colab's GPU runtime** (Tesla T4) for the actual training.
>
> **Lightweight Codebase Packaging:** To avoid uploading large local project folders to Colab, we have created a zipping script that packages the codebase into a clean, lightweight archive (excluding `venv/`, logs, and temporary files) and zips the dataset separately for fast uploads.

## Proposed Changes

We will package the components and instructions for the classification pipeline.

---

### Classification Component

#### [NEW] [zip_disease_classifier.py](file:///d:/KrishiMitra2/ai_models/disease_detection/scripts/zip_disease_classifier.py)
* Automatically packages the codebase into `outputs/disease_detection_code.zip` (excluding environments and large assets).
* Packages the raw PlantVillage images into `datasets/raw/plantvillage.zip` for direct upload or Drive sync.

#### [NEW] [Disease_Classification_Training.ipynb](file:///d:/KrishiMitra2/ai_models/disease_detection/notebooks/Disease_Classification_Training.ipynb)
* Colab notebook containing cells to unzip files, install dependencies, execute 30-epoch training on GPU, run test-set evaluation, and download the resulting `efficientnet_b0_disease.pt` weights and `class_mapping.json` files.

#### [NEW] [colab_classification_instructions.md](file:///d:/KrishiMitra2/ai_models/disease_detection/colab_classification_instructions.md)
* Markdown guide explaining how to pack local folders, upload files, configure runtime, and download results from Colab.

#### [MODIFY] [README.md](file:///d:/KrishiMitra2/ai_models/disease_detection/README.md)
* Update checklist status for Phase 4.

---

## Verification Plan

### Automated Tests
1. **Model Architecture Unit Test:** Verifies PyTorch classification model loading, forward pass, logit output shapes, and backpropagation on mock inputs.
   ```powershell
   $env:PYTHONPATH="."
   .\venv\Scripts\python.exe tests/test_classifier_model.py
   ```
2. **Dataset & Loader Verification:** Verifies that the dataset adapter and DataLoader factory can load PlantVillage CSV files and access image batches:
   ```powershell
   $env:PYTHONPATH="."
   .\venv\Scripts\python.exe data/dataset.py
   .\venv\Scripts\python.exe data/dataloader.py
   ```
3. **Colab Package Testing:** Verify that the classification packaging script generates the ZIP files successfully:
   ```powershell
   $env:PYTHONPATH="."
   .\venv\Scripts\python.exe scripts/zip_disease_classifier.py
   ```

### Manual Verification
1. User will upload the generated zip files and notebook to Google Colab, select GPU runtime, and execute training.
2. Verify that the model runs validation, achieves high accuracy, and successfully downloads the output weights.
