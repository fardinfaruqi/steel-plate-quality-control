# Steel Plate Defect Predictor — React Frontend

A bonus web interface for the steel plate quality-control data-mining project. It exposes the trained SVM (see the parent project `../README.md`) through a React app backed by a small FastAPI service.

The frontend is fully metadata-driven: it fetches the feature list, input groups, integer/binary constraints, and UI copy from the API, so nothing about the model layout is hardcoded.

## Architecture

```
React app (Vite + TypeScript + Tailwind)
        │  fetch("/api/...")  (dev proxy: /api → http://localhost:8000)
        ▼
FastAPI service (../api/main.py)
        │
        ▼
Reusable prediction module (../src/predict.py)
        │
        ▼
Saved artifacts (../outputs/models/*.pkl)
```

- `../api/main.py` — FastAPI service. Reuses `../src/predict.py`, so the exact production artifacts (`feature_names.pkl`, `svm_scaler.pkl`, `tuned_svm_model.pkl`) are used and predictions match the model exactly.
- `../src/predict.py` — model loading, feature ordering, scaling, validation, and prediction.
- The app never retrains the model.

## Prerequisites

- Python 3.14+ with a virtual environment at the project root (`.venv`).
- Node.js 20+ and npm.

## Setup

### 1. Python dependencies

From the project root, install the data-mining dependencies (includes pandas/scikit-learn/joblib used by the prediction module) and the API dependencies:

```
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -r api/requirements.txt
```

### 2. Frontend dependencies

From this directory:

```
npm install
```

## Running

### 1. Start the API (terminal 1, from the project root)

```
.venv\Scripts\python -m uvicorn api.main:app --port 8000
```

### 2. Start the dev server (terminal 2, from this directory)

```
npm run dev
```

Open `http://localhost:5173`. The Vite dev server proxies `/api` calls to the backend on port `8000` (see `vite.config.ts`), so no environment configuration is needed.

## Building for production

```
npm run build      # type-checks (tsc -b) and bundles to dist/
npm run preview    # serve the production build locally
```

## Linting

```
npm run lint       # oxlint
```

## API

Base URL: `http://localhost:8000`

| Endpoint | Method | Description |
|---|---|---|
| `/api/health` | GET | Service status. |
| `/api/metadata` | GET | Feature order, input groups, integer/binary features, model info, UI copy, and fault types. |
| `/api/examples` | GET | The 389 held-out test-set rows (the model never saw them during training) with their actual defect labels, in model feature order. |
| `/api/predict` | POST | Body `{"features": {<feature>: <value>, ...}}` → `{"prediction": "..."}`. Returns 400 if the feature set is invalid. |

Interactive docs are available at `http://localhost:8000/docs` while the API is running.

## Tests

Run the API contract tests from the project root:

```
.venv\Scripts\python -m pytest api/test_api.py
```

