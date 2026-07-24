# Implementation Plan - Phase 7: FastAPI Integration

This plan covers the design and execution of **Phase 7: FastAPI Integration**. We will implement a production-ready REST API server inside the `backend/` directory that exposes our AI Prediction Pipeline to the Flutter mobile application, logs prediction histories, and serves visual crop outputs.

## User Review Required

> [!IMPORTANT]
> **Database Configuration & Fallback:** We will implement ORM schemas using **SQLAlchemy**. By default, the application will attempt to connect to a **PostgreSQL** instance using the `DATABASE_URL` environment variable. If PostgreSQL is not active or configured, the app will gracefully fall back to a local **SQLite** database (`krishimitra.db`), allowing out-of-the-box local testing without manual database provisioning.
>
> **Static File Serving:** The server will mount the `outputs/pipeline` directories to serve original, annotated, and cropped images under static HTTP endpoints (e.g., `/static/val_00000_annotated.jpg`), which is required for display inside the Flutter mobile app.

## Proposed Changes

---

### Backend Server Structure

We will create the following layout under the `backend/` workspace folder:

```text
backend/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application initialization & routes mounting
│   ├── config.py        # Settings loader (DB URL, Static Paths, OOD Threshold)
│   ├── database.py      # SQLAlchemy engine & session configurations
│   ├── models.py        # PredictionRecord and LeafRecord ORM database models
│   ├── schemas.py       # Pydantic data schemas for requests and responses
│   ├── crud.py          # Create/Read database helper functions
│   └── routers/
│       ├── __init__.py
│       ├── predict.py   # POST /api/v1/predict (accepts upload, runs pipeline, logs to DB)
│       └── history.py   # GET /api/v1/history (queries past farmer diagnoses)
└── tests/
    └── test_server.py   # Integration tests for server endpoints
```

---

### API Endpoints

#### 1. System Health Check
* **`GET /health`** or **`GET /`**
* Returns status `{"status": "healthy"}`.

#### 2. Disease Prediction Endpoint
* **`POST /api/v1/predict`**
* Expects multipart form upload with field `file` (image).
* Accepts optional parameters: `threshold` (float, default: 0.60).
* Saves file, calls `PredictionPipeline.predict()`, saves results to PostgreSQL/SQLite, and returns the full JSON diagnosis report.

#### 3. Diagnosis History Endpoints
* **`GET /api/v1/history`**: Returns list of all past diagnosis records (paginated).
* **`GET /api/v1/history/{id}`**: Returns full detail (including leaf logs and prompt contexts) for a specific record ID.
* **`DELETE /api/v1/history/{id}`**: Deletes a record from history.

---

## Verification Plan

### Automated Tests
1. Run the FastAPI server unit/integration tests (using a temporary SQLite test database):
   ```powershell
   $env:PYTHONPATH="."
   .\venv\Scripts\python.exe -m unittest backend/tests/test_server.py
   ```
2. Start the FastAPI server locally:
   ```powershell
   $env:PYTHONPATH="."
   .\venv\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
   ```
3. Test prediction endpoint using a mock client (e.g. `curl` or a python test script):
   ```powershell
   # In another terminal window:
   .\venv\Scripts\python.exe -c "import requests; r = requests.post('http://127.0.0.1:8000/api/v1/predict', files={'file': open('ai_models/disease_detection/datasets/raw/plantdoc_detection/images/val/val_00000.jpg', 'rb')}); print(r.status_code); print(r.json()['leaves_found'])"
   ```
