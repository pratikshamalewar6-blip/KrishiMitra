# Task Checklist - KrishiMitra UI Redesign & Full System Integration

- [ ] **Task 1: Backend Multi-Language & Gemini Service Updates**
  - [ ] Add `lang` parameter (`en` / `hi`) to `gemini_service.py` for English/Hindi advice synthesis
  - [ ] Update `POST /api/v1/predict` router in `backend/app/routers/predict.py` to pass `lang` to Gemini generator

- [ ] **Task 2: Build `DiseaseDetectionScreen` in Flutter**
  - [ ] Create `mobile_app/lib/screens/disease_detection_screen.dart` with reference layout:
    - App Header: `Disease Detection 🛡️`, Subtitle, `History` button, `EN/HI` language toggle
    - Dismissable `💡 Tip: Capture clear image of affected leaf for better results` card
    - Camera Viewfinder Card with Flash `⚡`, Switch `📷`, Gallery button `🖼️`, Center Shutter `⚪`, How to Capture button `❓`
    - Horizontal scrollable "How to capture?" cards (Focus, Lighting, Single leaf, Camera steady, Avoid distant shots)
    - Recent Detections card with leaf thumbnail, disease title, risk chip (`Medium Risk`), date timestamp, and `View All >`
    - Early Detection Banner `🛡️`
    - Custom Bottom Navigation Bar (`Home`, `Crop`, `Scan Disease` elevated FAB, `Alerts (3)`, `Profile`)
  - [ ] Support camera capture & gallery image picking

- [ ] **Task 3: Update Main Entry Point & Theme in Flutter**
  - [ ] Update `mobile_app/lib/main.dart` to use `DiseaseDetectionScreen` as the home launcher
  - [ ] Configure light/emerald green theme matching design

- [ ] **Task 4: Connect API & Prediction Models**
  - [ ] Update `mobile_app/lib/services/api_service.dart` to pass language selection (`lang=hi` or `lang=en`)
  - [ ] Ensure `DiagnosticScreen` and `ResultsScreen` render localized Gemini advice

- [ ] **Task 5: IEEE Publication Paper & Thesis Documentation**
  - [ ] Write `docs/krishimitra_ieee_paper.md` detailing multi-stage architecture, quantitative metrics ($mAP = 0.9378$), OOD thresholding, and real-world results

- [ ] **Task 6: System Launch & Verification**
  - [ ] Run backend tests to verify language support
  - [ ] Test full pipeline and provide exact launch commands for backend and Flutter app
