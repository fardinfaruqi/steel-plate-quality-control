"""Regression tests for the FastAPI backend (api/main.py).

Run from the project root with:
    .venv\\Scripts\\python.exe -m pytest api/test_api.py
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app, load_examples, FAULT_COLUMNS

client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_metadata_contract():
    res = client.get("/api/metadata")
    assert res.status_code == 200
    meta = res.json()
    assert len(meta["feature_order"]) == 27
    assert len(meta["groups"]) == 5
    assert sum(len(g["features"]) for g in meta["groups"]) == 27
    assert set(meta["binary_features"]) == {"TypeOfSteel_A300", "TypeOfSteel_A400"}
    assert set(meta["fault_types"]) == set(FAULT_COLUMNS)
    assert meta["model_info"]["Final Model"] == "RBF SVM"
    assert meta["disclaimer"]


def test_examples_returns_all_rows_and_every_defect_type():
    res = client.get("/api/examples")
    assert res.status_code == 200
    body = res.json()
    assert body["total"] == 1941
    assert len(body["rows"]) == 1941
    assert body["feature_order"] == client.get("/api/metadata").json()["feature_order"]
    defects = {row["defect"] for row in body["rows"]}
    assert defects == set(FAULT_COLUMNS)
    first = body["rows"][0]
    assert first["index"] == 0
    assert len(first["values"]) == 27


def test_predict_known_example():
    X, y, feature_order = load_examples()
    payload = {
        feature: float(X.iloc[0][feature])
        for feature in feature_order
    }
    res = client.post("/api/predict", json={"features": payload})
    assert res.status_code == 200
    assert res.json()["prediction"] == "Other_Faults"


def test_predict_reorders_features():
    X, y, feature_order = load_examples()
    row = X.iloc[5]
    shuffled = {f: float(row[f]) for f in reversed(feature_order)}
    res = client.post("/api/predict", json={"features": shuffled})
    assert res.status_code == 200
    assert res.json()["prediction"] == y.iloc[5]


def test_predict_missing_feature_returns_400():
    X, y, feature_order = load_examples()
    payload = {f: float(X.iloc[0][f]) for f in feature_order[:-1]}
    res = client.post("/api/predict", json={"features": payload})
    assert res.status_code == 400
    assert "Missing required feature" in res.json()["detail"]
