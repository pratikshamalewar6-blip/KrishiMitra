# Implementation Plan - Gemini AI Explanation Synthesis

This plan covers the design and execution of integrating **Gemini AI API explanations** into our prediction pipeline. The goal is to synthesize the verified agricultural facts retrieved from the Knowledge Base into friendly, reassuring, and structurally clear advice for the farmer, while strictly preventing hallucination of unverified organic/chemical remedies.

## User Review Required

> [!IMPORTANT]
> **API Key Configuration:** The backend will load `GEMINI_API_KEY` from the system environment variables. If the key is not set, the server will gracefully operate in **Mock Fallback Mode** (generating a mock advice string based directly on the database context) to prevent crashes and allow out-of-the-box local testing.
>
> **Hallucination Prevention:** The prompt template will explicitly instruct the Gemini model to use *only* the organic/chemical controls, fertilizers, and preventions provided in the database context. Gemini will be strictly forbidden from suggesting unverified or external home remedies or chemical pesticides.

## Proposed Changes

---

### 1. Database & Schema Updates

#### [MODIFY] [models.py](file:///d:/KrishiMitra2/backend/app/models.py)
* Add a new column `gemini_explanation` (String, nullable=True) to the `LeafRecord` model.

#### [MODIFY] [schemas.py](file:///d:/KrishiMitra2/backend/app/schemas.py)
* Update `LeafRecordBase` to include `gemini_explanation: Optional[str] = None`.

#### [MODIFY] [crud.py](file:///d:/KrishiMitra2/backend/app/crud.py)
* Map `gemini_explanation` when writing new `LeafRecord` records to the database.

---

### 2. Gemini AI Explanation Generator

#### [NEW] [gemini_service.py](file:///d:/KrishiMitra2/backend/app/services/gemini_service.py)
* Implements `GeminiExplanationGenerator` class:
  * Initializes the `google-generativeai` client using `GEMINI_API_KEY`.
  * Checks for API key availability. If missing, automatically falls back to static template rendering (Mock Mode).
  * Method `generate_explanation(prompt_context: dict) -> str`:
    * Constructs a constrained prompt containing the crop, disease, severity, symptoms, organic remedies, chemical remedies, weather conditions, and prevention rules.
    * Calls the Gemini API (`gemini-1.5-flash` or similar) to synthesize the response.
    * Post-processes and returns the cleaned markdown advice text.

#### [MODIFY] [predict.py](file:///d:/KrishiMitra2/backend/app/routers/predict.py)
* Import `GeminiExplanationGenerator` and initialize it.
* Inside `upload_and_predict`, for each leaf result, call the generator using `prompt_context` and save the output text in `gemini_explanation` before database commit.

---

### 3. Flutter Frontend Update

#### [MODIFY] [prediction.dart](file:///d:/KrishiMitra2/mobile_app/lib/models/prediction.dart)
* Add `geminiExplanation` string property to `LeafRecord` model.

#### [MODIFY] [results_screen.dart](file:///d:/KrishiMitra2/mobile_app/lib/screens/results_screen.dart)
* Add a new tab/card/section titled **"KrishiMitra AI Assistant"** displaying the generated markdown Gemini advice.

---

## Verification Plan

### Automated Tests
1. Run server tests to confirm that schema changes load and run successfully:
   ```powershell
   $env:PYTHONPATH="ai_models/disease_detection"
   .\ai_models\disease_detection\venv\Scripts\python.exe backend/tests/test_server.py
   ```
2. Test prediction route manually using a test script to print the generated Gemini explanation:
   ```powershell
   # In PowerShell:
   $env:PYTHONPATH="ai_models/disease_detection"
   .\ai_models\disease_detection\venv\Scripts\python.exe -c "from backend.app.services.gemini_service import GeminiExplanationGenerator; generator = GeminiExplanationGenerator(); print(generator.generate_explanation({'crop': 'Tomato', 'disease': 'Late blight', 'confidence': 90.0, 'symptoms': ['Spots on leaves'], 'organic_treatment': ['Copper sprays'], 'chemical_treatment': ['Fungicides'], 'prevention': ['Crop rotation'], 'severity': 'High'}))"
   ```
