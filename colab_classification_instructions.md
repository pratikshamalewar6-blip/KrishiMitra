# 🚀 Step-by-Step Training Guide: EfficientNet-B0 Disease Classifier on Google Colab

We have simplified the training process! You **no longer need to upload or zip the large 2 GB dataset** from your local computer. Instead, you can download the PlantVillage dataset directly inside Google Colab.

You only need to upload the lightweight codebase zip file (**`outputs/disease_detection_code.zip`**, ~1.7 MB).

---

## 📦 Step 1: Run the Packaging Helper Script
Locally, run the zipping script to package your codebase files:
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\python.exe scripts/zip_disease_classifier.py
```
This will generate:
* **Lightweight Codebase:** `outputs/disease_detection_code.zip` (Size: **~1.7 MB**)

---

## 🌐 Step 2: Open and Configure Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **Upload** and upload the local Jupyter Notebook file:
   * [Disease_Classification_Training.ipynb](file:///d:/KrishiMitra2/ai_models/disease_detection/notebooks/Disease_Classification_Training.ipynb)
3. Set your runtime type to **GPU** for accelerated training:
   * In the top menu, go to **Runtime** > **Change runtime type**.
   * Select **T4 GPU** (or any other available GPU accelerator) and click **Save**.

---

## 📤 Step 3: Upload the Codebase and Crops Zips
1. Open the left sidebar in Colab (folder icon).
2. Drag and drop these files from your PC directly into the Colab storage panel (to `/content/`):
   * **`outputs/disease_detection_code.zip`** (Size: ~2.5 MB)
   * **`outputs/plantdoc_crops.zip`** (Size: ~68.5 MB)

---

## 🏋️ Step 4: Run Training in the Notebook
Run the notebook cells sequentially:
1. **Unzip codebase & crops:** Extracts python scripts and preprocessed real-world segmented crops.
2. **Download dataset:** Downloads and extracts the PlantVillage dataset directly in Colab (takes ~1-2 minutes).
3. **Install dependencies:** Installs requirements.
4. **Train Base Classifier:** Trains on standard PlantVillage.
   ```bash
   !PYTHONPATH=. python classification/train_classifier.py --epochs 30 --batch-size 32 --device cuda --dataset plantvillage
   ```
5. **Train Real-World Classifier:** Runs the two-stage fine-tuning process on the mixed dataset (PlantVillage + PlantDoc crops) with Albumentations, class weights, and differential learning rates:
   ```bash
   !PYTHONPATH=. python classification/train_realworld.py --epochs 20 --stage1-epochs 5 --batch-size 32 --mix-ratio 0.15 --loss WeightedCrossEntropy --device cuda
   ```
6. **Evaluate & Compare Models:** Evaluates both weights separately on PlantVillage and PlantDoc test splits, generating the comparative report:
   ```bash
   !PYTHONPATH=. python classification/evaluate_realworld.py
   ```

---

## 💾 Step 5: Place Trained Weights Locally
Once the browser downloads are finished, copy the files back into your local repository workspace:
1. Move the downloaded weights to:
   * **`saved_models/efficientnet_b0_disease.pt`** (Base Model)
   * **`saved_models/efficientnet_b0_realworld.pt`** (Improved Real-World Model)
2. Move the downloaded class mapping and comparison report to:
   * **`outputs/classification/class_mapping.json`**
   * **`outputs/classification/realworld_comparison_report.md`**

---

## 📸 Step 6: Test Locally on Segmented Crops
To test predictions locally using the improved model with OOD and Top-5 output format, run:
```powershell
$env:PYTHONPATH="."
.\venv\Scripts\python.exe classification/predict_classifier.py --image outputs/segmentations/val_00000/leaf_segmented_1.png --weights saved_models/efficientnet_b0_realworld.pt
```
This will print the diagnosis, confidence, and Top-5 calibration breakdown.

