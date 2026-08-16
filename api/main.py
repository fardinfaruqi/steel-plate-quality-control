"""FastAPI backend for the React Steel Plate Defect Predictor.

Exposes the trained SVM as a small HTTP API consumed by the React
frontend in ``frontend/``. Model loading, feature ordering, scaling and
prediction are delegated to the reusable ``src.predict`` module so the
exact production artifacts are reused.

Run from the project root with:
    uvicorn api.main:app --reload --port 8000
"""

import sys
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.predict import get_feature_names, predict_defect

# ---------------------------------------------------------------------------
# Dataset / constants (mirrors the feature definitions used across the app)
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "Faults.NNA"
COLUMNS_PATH = DATA_DIR / "Faults27x7_var"

FAULT_COLUMNS = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains",
    "Dirtiness", "Bumps", "Other_Faults",
]

BINARY_FEATURES = {"TypeOfSteel_A300", "TypeOfSteel_A400"}

INTEGER_FEATURES = {
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity",
    "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Length_of_Conveyer",
    "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness",
}

FEATURE_GROUPS = [
    ("Position", [
        "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum",
    ]),
    ("Geometric Measurements", [
        "Pixels_Areas", "X_Perimeter", "Y_Perimeter", "Outside_X_Index",
        "Edges_X_Index", "Edges_Y_Index", "Square_Index",
        "Orientation_Index",
    ]),
    ("Luminosity", [
        "Sum_of_Luminosity", "Minimum_of_Luminosity", "Maximum_of_Luminosity",
        "Luminosity_Index",
    ]),
    ("Conveyor / Steel Information", [
        "Length_of_Conveyer", "TypeOfSteel_A300", "TypeOfSteel_A400",
        "Steel_Plate_Thickness",
    ]),
    ("Shape / Edge / Area Features", [
        "Edges_Index", "Empty_Index", "Outside_Global_Index", "LogOfAreas",
        "Log_X_Index", "Log_Y_Index", "SigmoidOfAreas",
    ]),
]

MODEL_INFO = {
    "Final Model": "RBF SVM",
    "C": "10",
    "Number of Features": "27",
    "Test Accuracy": "76.35%",
    "Test F1-Score": "76.44%",
}

ABOUT_MODEL = {
    "Model": "Tuned Support Vector Machine",
    "Kernel": "RBF",
    "C": "10",
    "Number of input features": "27",
    "Test Accuracy": "76.35%",
    "Test F1-Score": "76.44%",
}

DISCLAIMER = (
    "This application provides a machine-learning prediction based on the "
    "trained dataset. It should be considered a decision-support tool and "
    "should not replace physical inspection or established industrial "
    "quality-control procedures."
)

NO_PROBABILITY_NOTE = (
    "No probability/confidence score is displayed because the trained SVC "
    "does not support probability prediction."
)


def group_slug(group_name):
    """Key-safe slug used for the measurement panel container."""
    return group_name.strip().lower().replace(" / ", "-").replace(" ", "-")


@lru_cache(maxsize=1)
def load_examples():
    """Return (X, y, feature_order) for ALL dataset rows.

    X columns are reordered to the model's feature order. y holds the
    actual defect type, used only to label examples for reference; it is
    never used as model input.
    """
    with open(COLUMNS_PATH, "r") as f:
        columns = [line.strip() for line in f]
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=columns)
    df["Defect_Type"] = df[FAULT_COLUMNS].idxmax(axis=1)
    df = df.drop(columns=FAULT_COLUMNS)
    X = df.drop(columns=["Defect_Type"])
    y = df["Defect_Type"]
    feature_order = get_feature_names()
    X = X[feature_order]
    return X, y, feature_order


app = FastAPI(title="Steel Plate Defect Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: dict[str, float]


router = APIRouter(prefix="/api")


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/metadata")
def metadata():
    feature_names = get_feature_names()
    return {
        "feature_order": feature_names,
        "groups": [
            {
                "name": name,
                "slug": group_slug(name),
                "features": [f for f in features if f in feature_names],
            }
            for name, features in FEATURE_GROUPS
        ],
        "integer_features": sorted(INTEGER_FEATURES),
        "binary_features": sorted(BINARY_FEATURES),
        "model_info": MODEL_INFO,
        "about_model": ABOUT_MODEL,
        "disclaimer": DISCLAIMER,
        "no_probability_note": NO_PROBABILITY_NOTE,
        "fault_types": list(FAULT_COLUMNS),
    }


@router.get("/examples")
def examples():
    X, y, feature_order = load_examples()
    return {
        "total": len(X),
        "feature_order": feature_order,
        "rows": [
            {
                "index": i,
                "defect": y.iloc[i],
                "values": X.iloc[i].tolist(),
            }
            for i in range(len(X))
        ],
    }


@router.post("/predict")
def predict(request: PredictRequest):
    try:
        input_df = pd.DataFrame([request.features])
        prediction = predict_defect(input_df)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"prediction": prediction}


app.include_router(router)
