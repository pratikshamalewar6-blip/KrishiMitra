# Implementation Plan - KrishiMitra UI Redesign & All Options Integration

We will implement the exact **Disease Detection Screen** matching the reference UI image provided by the user, and execute all 4 requested expansion options:
1. **Option 1:** System Launch (FastAPI Backend + Flutter App)
2. **Option 2:** Live Gemini AI API configuration with fallback
3. **Option 3:** Multi-Language support (English & Hindi) in Flutter & Gemini prompts
4. **Option 4:** Major Project Thesis / IEEE Paper Documentation

---

## User Review Required

> [!IMPORTANT]
> **Pixel-Perfect Screen Redesign:** We will build `DiseaseDetectionScreen` matching every visual detail of the reference screenshot:
> - Header with shield badge `🛡️` and `History` action.
> - Dismissable `💡 Tip: Capture clear image of affected leaf for better results` banner.
> - Camera Viewfinder Card with overlay Flash `⚡`, Switch `📷`, **Gallery** `🖼️`, **Center Shutter Button** `⚪`, and **How to capture** `❓`.
> - **How to capture?** horizontal scrollable cards (Focus, Lighting, Single leaf, Camera steady, Avoid distant shots).
> - **Recent Detections** card with leaf thumbnail, risk level badge (e.g. `Medium Risk`), date timestamp, and `View All >`.
> - **Early Detection Shield Banner**.
> - **Custom Bottom Navigation Bar** (`Home`, `Crop`, `Scan Disease` elevated center button, `Alerts (3)`, `Profile`).
> - Integrated **Language Switcher** toggle (`EN` / `HI`) in the top app bar to demonstrate Option 3!

---

## Proposed Changes

### 1. Flutter Mobile Application

#### [NEW] [disease_detection_screen.dart](file:///d:/KrishiMitra2/mobile_app/lib/screens/disease_detection_screen.dart)
* Implements the exact UI design matching the provided screenshot.
* Integrates `ImagePicker` for camera and gallery triggers.
* Displays past prediction logs with dynamic risk chips (`Low Risk`, `Medium Risk`, `High Risk`).
* Includes horizontal scrollable "How to capture?" cards.
* Includes dismissable Tip Banner and Early Detection shield footer.
* Features the custom 5-tab Bottom Navigation bar with elevated center "Scan Disease" FAB.
* Includes language translation dictionary (English & Hindi).

#### [MODIFY] [main.dart](file:///d:/KrishiMitra2/mobile_app/lib/main.dart)
* Sets `DiseaseDetectionScreen` as the default home launcher screen.
* Updates light/emerald green theme tokens (`#1E824C`, `#2ECC71`, `#E8F5E9`, `#0A3622`).

#### [MODIFY] [api_service.dart](file:///d:/KrishiMitra2/mobile_app/lib/services/api_service.dart) & [prediction.dart](file:///d:/KrishiMitra2/mobile_app/lib/models/prediction.dart)
* Add support for passing `language` query parameter (`en` or `hi`) to backend prediction endpoints.

---

### 2. Backend & Gemini AI Service

#### [MODIFY] [gemini_service.py](file:///d:/KrishiMitra2/backend/app/services/gemini_service.py)
* Add `lang` support (`hi` for Hindi, `en` for English) to `generate_explanation`.
* When `lang='hi'`, Gemini generates localized Hindi farmer advice (or localized mock advice if offline).

#### [MODIFY] [predict.py](file:///d:/KrishiMitra2/backend/app/routers/predict.py)
* Accept optional `lang` query parameter in `POST /api/v1/predict?lang=hi`.

---

### 3. Project Documentation & IEEE Thesis Artifact

#### [NEW] [krishimitra_ieee_paper.md](file:///d:/KrishiMitra2/docs/krishimitra_ieee_paper.md)
* Comprehensive research thesis & IEEE publication draft summarizing:
  - Abstract & Introduction
  - Multi-stage Architecture (YOLOv11 → SAM2 → EfficientNet-B0 → Knowledge Base → Gemini AI)
  - Quantitative Experimental Results ($mAP_{50} = 0.9378$, Precision, Recall)
  - Real-World Generalization & Out-of-Distribution (OOD) Confidence Thresholding

---

## Verification Plan

### Automated & Unit Verification
1. Test updated backend prediction with language parameter:
   ```powershell
   $env:PYTHONPATH="ai_models/disease_detection"
   .\ai_models\disease_detection\venv\Scripts\python.exe -c "from backend.app.services.gemini_service import GeminiExplanationGenerator; g = GeminiExplanationGenerator(); print(g.generate_explanation({'crop': 'Wheat', 'disease': 'Yellow Rust', 'confidence': 94.5, 'symptoms': ['Yellow rust pustules'], 'organic_treatment': ['Neem oil spray'], 'chemical_treatment': ['Propiconazole'], 'prevention': ['Use resistant varieties'], 'severity': 'Medium'}, lang='hi'))"
   ```

2. Run backend test suite:
   ```powershell
   .\ai_models\disease_detection\venv\Scripts\python.exe -m unittest discover -s backend/tests
   ```

### Manual Visual Verification
1. Launch Flutter app and verify all UI components (Tip banner, camera viewfinder card, horizontal tips, recent detections card, bottom nav bar).
2. Test camera and gallery pick flow to verify live diagnostic loading and results screens.
