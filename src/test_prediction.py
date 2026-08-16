"""Test the reusable prediction module (src/predict.py).

Uses real observations from the original dataset (without the target
column) to verify that the module:
  - predicts from the 27 real features
  - is insensitive to shuffled feature-column order
  - raises clear errors for missing / extra / non-numeric input

Run with:  python src/test_prediction.py
"""

from pathlib import Path

import joblib
import pandas as pd

from data import load_plate_data
from predict import predict_defect, predict_defects

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

FEATURES_PATH = MODELS_DIR / "feature_names.pkl"


def report(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -> {detail}" if detail else ""))


def main():
    feature_names = joblib.load(FEATURES_PATH)
    X, y = load_plate_data()

    print("=" * 72)
    print("TEST - REAL OBSERVATION (main path)")
    print("=" * 72)
    for idx in (0, 5, 42):
        obs = X.iloc[[idx]]
        actual = y.iloc[idx]
        predicted = predict_defect(obs)
        print(f"  observation {idx:>3} | Actual: {actual:<12} | "
              f"Predicted: {predicted}")

    obs = X.iloc[[0]]
    actual = y.iloc[0]
    predicted = predict_defect(obs)
    print()
    print("Actual Defect:  ", actual)
    print("Predicted Defect:", predicted)

    print()
    print("=" * 72)
    print("TEST 1 - CORRECT 27-FEATURE INPUT")
    print("=" * 72)
    try:
        pred = predict_defect(X.iloc[[7]])
        report("single-row input", isinstance(pred, str) and len(pred) > 0,
               f"predicted={pred}")
    except Exception as e:
        report("single-row input", False, f"{type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print("TEST 2 - SHUFFLED FEATURE COLUMN ORDER")
    print("=" * 72)
    shuffled = X.iloc[[7]][list(reversed(X.columns))]
    pred_orig = predict_defect(X.iloc[[7]])
    pred_shuf = predict_defect(shuffled)
    report("shuffled order gives same prediction",
           pred_orig == pred_shuf,
           f"{pred_orig} vs {pred_shuf}")

    print()
    print("=" * 72)
    print("TEST 3 - MISSING FEATURE")
    print("=" * 72)
    try:
        predict_defect(X.iloc[[7]].drop(columns=["Luminosity_Index"]))
        report("missing feature raises ValueError", False, "no error raised")
    except ValueError as e:
        report("missing feature raises ValueError", True, str(e))
    except Exception as e:
        report("missing feature raises ValueError", False,
               f"wrong error type {type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print("TEST 4 - EXTRA UNEXPECTED FEATURE")
    print("=" * 72)
    bad = X.iloc[[7]].copy()
    bad["Not_A_Feature"] = 1.0
    try:
        predict_defect(bad)
        report("extra feature raises ValueError", False, "no error raised")
    except ValueError as e:
        report("extra feature raises ValueError", True, str(e))
    except Exception as e:
        report("extra feature raises ValueError", False,
               f"wrong error type {type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print("TEST 5 - NON-NUMERIC FEATURE VALUE")
    print("=" * 72)
    bad = X.iloc[[7]].copy()
    bad["X_Minimum"] = "abc"
    try:
        predict_defect(bad)
        report("non-numeric value raises ValueError", False, "no error raised")
    except ValueError as e:
        report("non-numeric value raises ValueError", True, str(e))
    except Exception as e:
        report("non-numeric value raises ValueError", False,
               f"wrong error type {type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print("TEST 6 - MISSING VALUE (NaN)")
    print("=" * 72)
    bad = X.iloc[[7]].copy()
    bad.iloc[0, 0] = float("nan")
    try:
        predict_defect(bad)
        report("NaN value raises ValueError", False, "no error raised")
    except ValueError as e:
        report("NaN value raises ValueError", True, str(e))
    except Exception as e:
        report("NaN value raises ValueError", False,
               f"wrong error type {type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print("TEST 7 - BATCH PREDICTION (predict_defects)")
    print("=" * 72)
    batch = X.iloc[[0, 1, 2]]
    series = predict_defects(batch)
    report("batch returns Series", isinstance(series, pd.Series),
           f"shape={series.shape}")
    report("batch predictions align to rows", list(series.index) == [0, 1, 2])
    print("  batch results:")
    for i, (actual_row, pred) in enumerate(zip(y.iloc[[0, 1, 2]].tolist(), series.tolist())):
        print(f"    row {i}: actual={actual_row:<12} predicted={pred}")

    print()
    print("=" * 72)
    print("TEST 8 - INPUT TYPE VALIDATION")
    print("=" * 72)
    try:
        predict_defect(X.iloc[[7]].values)
        report("non-DataFrame input raises TypeError", False, "no error raised")
    except TypeError as e:
        report("non-DataFrame input raises TypeError", True, str(e))
    except Exception as e:
        report("non-DataFrame input raises TypeError", False,
               f"wrong error type {type(e).__name__}: {e}")

    print()
    print("=" * 72)
    print("ALL TESTS COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
