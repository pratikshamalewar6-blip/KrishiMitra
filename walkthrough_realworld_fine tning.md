# Real-World Farm Generalization & Crop Guard Walkthrough

We have implemented a 3-step solution to solve real-world farm generalization and cross-crop misclassifications:

---

## 1. Crop Selection Guard (Backend & Pipeline)
* **`prediction_pipeline.py`**: Added `crop_hint` parameter support to `_predict_disease_pil()` and `predict()`. When specified (e.g., `crop_hint="Tomato"` or `crop_hint="Potato"`), probabilities are masked to that crop's disease subspace, **eliminating 100% of wrong-crop confusion**.
* **`backend/app/routers/predict.py`**: Added `crop_hint` query parameter to the `POST /api/v1/predict` FastAPI endpoint.
* **`mobile_app/lib/services/api_service.dart`**: Updated `predictImage()` to pass `crop_hint` to the backend when selected by the user.

---

## 2. Notebook 3 for Google Colab (`3_RealWorld_Field_FineTuning.ipynb`)
* **File Path**: [3_RealWorld_Field_FineTuning.ipynb](file:///d:/KrishiMitra2/notebooks/3_RealWorld_Field_FineTuning.ipynb)
* **Features**:
  - Heavy **Albumentations Field-Noise Augmentations** (CoarseDropout/Cutout, ShiftScaleRotate, MotionBlur, ColorJitter).
  - 2-stage fine-tuning (Head training + Backbone fine-tuning).
  - Automatically exports `efficientnet_b0_disease.pt` and `class_mapping.json`.

---

## 3. Top-3 Calibrated Predictions
* Pipeline logs and structures Top-3 ranked diagnoses with confidence breakdown.
