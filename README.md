# 🌱 KrishiMitra - AI Powered Crop Disease Detection

## Overview

KrishiMitra is an enterprise-grade AI-powered Smart Agriculture Platform designed to assist farmers, researchers, and agricultural organizations through intelligent crop disease detection and decision support.

This module focuses on real-world crop disease detection using a multi-stage computer vision pipeline.

---

## AI Pipeline

```text
       Farmer Uploads Image
                 │
                 ▼
       ─────────────────────────────
       1. Image Validation
       ─────────────────────────────
                 │
                 ▼
       ─────────────────────────────
       2. YOLOv11 Leaf Detection
       ─────────────────────────────
                 │
                 ▼
       Detected Leaf Bounding Box
                 │
                 ▼
       ─────────────────────────────
       3. SAM2 Leaf Segmentation
       ─────────────────────────────
                 │
                 ▼
         Background Removed Leaf
                 │
                 ▼
       ─────────────────────────────
       4. Image Preprocessing
          224 × 224, Normalization
       ─────────────────────────────
                 │
                 ▼
       ─────────────────────────────
       5. EfficientNet-B0
          Disease Classification
       ─────────────────────────────
                 │
                 ▼
          Disease + Confidence
                 │
                 ▼
       ─────────────────────────────
       6. Knowledge Base
       ─────────────────────────────
                 │
                 ▼
          • Symptoms          • Causes
          • Organic Treatment • Chemical Treatment
          • Fertilizer        • Pesticide
          • Prevention        • Severity
                 │
                 ▼
       ─────────────────────────────
       7. Gemini AI
          Generate Explanation
       ─────────────────────────────
                 │
                 ▼
       ─────────────────────────────
       8. FastAPI Backend
       ─────────────────────────────
                 │
                 ▼
       ─────────────────────────────
       9. Flutter App
       ─────────────────────────────
```

---

## Project Goals

- Production-ready AI architecture
- Research-quality implementation
- IEEE publication support
- Final year major project
- Startup-ready MVP
- Modular and extensible design

---

## Technology Stack

- Python
- PyTorch
- Ultralytics YOLOv11
- Meta SAM2
- EfficientNet-B0
- OpenCV
- Albumentations
- FastAPI
- PostgreSQL
- Flutter

---

## Project Status

- [x] Phase 0 - Architecture
- [x] Phase 1 - Dataset Verification
- [x] Phase 2 - YOLOv11 Leaf Detection
- [x] Phase 3 - SAM2 Segmentation
- [x] Phase 4 - EfficientNet-B0 Classification
- [x] Phase 4.5 - Real-World Classifier Generalization
- [x] Phase 4.6 - Real-World Validation
- [x] Phase 5 - Knowledge Base
- [x] Phase 6 - AI Prediction Pipeline
- [x] Phase 7 - FastAPI Integration
- [x] Phase 8 - Flutter Integration

---

## 🛡️ Production & Real-World Safeguards

- **Crop Selection Guard:** Eliminates cross-crop misclassification by enforcing user-selected crop context.
- **Confidence Thresholding:** Rejects ambiguous or unclear field captures to prevent false positives.
- **Field-Noise Generalization:** Trained with heavy Albumentations (Cutout, MotionBlur, ColorJitter) to withstand real farm lighting, shadows, and background soil noise.

