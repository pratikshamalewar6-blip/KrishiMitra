# Real-World Farm Generalization & Crop Guard Implementation Plan

This plan addresses cross-crop misclassification and domain-shift on real field images through a 3-step solution:
1. **Crop Selection Guard (Backend & Pipeline)**: Allows optional crop filtering (e.g., Tomato, Potato, Apple, Corn) to eliminate 100% of cross-crop misclassifications while keeping "Auto-Detect" mode available.
2. **Notebook 3 (Real-World Field Fine-Tuning)**: Fine-tunes EfficientNet-B0 on a balanced 50/50 mix of PlantVillage + PlantDoc real field images with heavy Albumentations field-noise augmentations (Cutout, Hue Shift, Contrast, Blur) so the model ignores soil and background foliage.
3. **Top-3 Calibrated Predictions (Backend & Web UI)**: Provides top-3 diagnostic confidence rankings for farmers when confidence is split across similar disease lesions.

---

## User Review Required

> [!IMPORTANT]
> **Key Improvement**: The **Crop Selection Guard** guarantees that if a farmer selects "Tomato", the AI will *only* diagnose among Tomato diseases, completely eliminating wrong-crop errors (like calling a Tomato leaf "Grape Black Rot").
> "Auto-Detect" will remain available for unguided detection.

---

## Proposed Changes

### Component 1: Prediction Pipeline & Backend (Crop Guard & Top-3)

#### [MODIFY] [prediction_pipeline.py](file:///d:/KrishiMitra2/ai_models/disease_detection/pipeline/prediction_pipeline.py)
- Add `crop_hint` parameter to `predict()` method.
- Implement crop filtering logic so logits are constrained to the target crop when specified.
- Include Top-3 prediction list in the result payload.

#### [MODIFY] [predict.py](file:///d:/KrishiMitra2/backend/app/routers/predict.py)
- Accept optional `crop_hint` query parameter in `POST /api/v1/predict`.

---

### Component 2: Notebook 3 for Google Colab (Real-World Field Fine-Tuning)

#### [NEW] [3_RealWorld_Field_FineTuning.ipynb](file:///d:/KrishiMitra2/notebooks/3_RealWorld_Field_FineTuning.ipynb)
- Self-contained Colab notebook for **50/50 Merged Field Training** (PlantVillage + PlantDoc real farm images).
- Includes heavy **Albumentations field augmentations** (Cutout/CoarseDropout, RandomBrightnessContrast, ShiftScaleRotate, MotionBlur).
- 2-Stage training with **Focal Loss** / **Weighted CrossEntropy**.

---

### Component 3: Mobile App & Web UI (Crop Selector & Top-3 Breakdown)

#### [MODIFY] [api_service.dart](file:///d:/KrishiMitra2/mobile_app/lib/services/api_service.dart)
- Pass `crop_hint` query parameter to backend.

---

## Verification Plan

### Automated Verification
- Run local pipeline tests (`test_pipeline.py` / `python -m uvicorn backend.app.main:app`).
- Verify crop filtering correctly constrains predictions when `crop_hint="Tomato"` or `crop_hint="Potato"`.

### Manual Verification
- Test sample real farm images in Google Chrome (both with Crop Selection Guard and Auto-Detect).
