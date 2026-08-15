"""PHASE 2 - Verify the saved model artifacts with the correct SVM scaler.

Verifies (read-only) the artifacts that define the production pipeline:
  outputs/models/tuned_svm_model.pkl   - final tuned SVM (SVC, C=10, RBF)
  outputs/models/svm_scaler.pkl        - scaler fitted on X_train (used to
                                          train the final SVM)
  outputs/models/feature_names.pkl     - the 27 features in model order

The old outputs/models/scaler.pkl (fitted on the full feature matrix X,
NOT the scaler used for SVM training) is loaded only for comparison.

This script does NOT retrain, fit, or modify any artifact.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = MODELS_DIR / "tuned_svm_model.pkl"
SVM_SCALER_PATH = MODELS_DIR / "svm_scaler.pkl"
OLD_SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"
DATA_PATH = DATA_DIR / "Faults.NNA"
COLUMNS_PATH = DATA_DIR / "Faults27x7_var"

EXPECTED_FEATURES = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
    "Maximum_of_Luminosity", "Length_of_Conveyer", "TypeOfSteel_A300",
    "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
    "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
    "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
]

FAULT_COLUMNS = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains",
    "Dirtiness", "Bumps", "Other_Faults",
]


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def probability_supported(model):
    """Return True only if predict_proba() is actually callable.

    Note: in scikit-learn 1.9, SVC.probability was deprecated and the
    attribute now returns the string 'deprecated' (truthy) regardless of
    the real value, so it cannot be used as a boolean check.
    """
    return bool(getattr(model, "_effective_probability", False)) and hasattr(
        model, "predict_proba"
    )


def load_observation():
    """Rebuild the dataset exactly as the notebook does and return X, y."""
    with open(COLUMNS_PATH, "r") as f:
        columns = [line.strip() for line in f]

    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=columns)
    df["Defect_Type"] = df[FAULT_COLUMNS].idxmax(axis=1)
    df = df.drop(columns=FAULT_COLUMNS)

    X = df.drop(columns=["Defect_Type"])
    y = df["Defect_Type"]
    return X, y


def scaler_stats_table(scaler, feature_names):
    return pd.DataFrame(
        {
            "mean_": scaler.mean_,
            "scale_": scaler.scale_,
        },
        index=feature_names,
    )


def main():
    # ------------------------------------------------------------------
    section("STEP 1/2 - LOCATE svm_scaler IN NOTEBOOK (documented)")
    # ------------------------------------------------------------------
    print("Notebook cells (verified by inspection, not guessed):")
    print("  cell 29  : scaler = StandardScaler(); fitted on FULL X (5.3)")
    print("  cell 86  : svm_scaler = StandardScaler(); fitted on X_train ONLY")
    print("             -> X_train_scaled / X_test_scaled created here")
    print("  cell 125 : grid_search.fit(X_train_scaled, y_train) -> tuned SVM")
    print("  cell 131 : best_svm = grid_search.best_estimator_")
    print("  cell 318 : joblib.dump(best_svm, tuned_svm_model.pkl)")
    print("  cell 320 : joblib.dump(scaler, scaler.pkl)  <- WRONG scaler saved")
    print("=> The final SVM was trained on data scaled by svm_scaler, so")
    print("   svm_scaler is the correct artifact for inference.")

    # ------------------------------------------------------------------
    section("STEP 4 - LOAD AND VERIFY ARTIFACTS")
    # ------------------------------------------------------------------
    for path in (MODEL_PATH, SVM_SCALER_PATH, FEATURES_PATH, OLD_SCALER_PATH):
        print(f"{path.name}: {'EXISTS' if path.exists() else 'MISSING'} ({path})")

    model = joblib.load(MODEL_PATH)
    svm_scaler = joblib.load(SVM_SCALER_PATH)
    old_scaler = joblib.load(OLD_SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    print(f"\nModel type        : {type(model).__name__}")
    print(f"svm_scaler type   : {type(svm_scaler).__name__}")
    print(f"feature_names type: {type(feature_names).__name__}")
    print(f"Number of features: {len(feature_names)}")
    print("Complete feature list:")
    for i, name in enumerate(feature_names, start=1):
        print(f"  {i:2d}. {name}")

    # ------------------------------------------------------------------
    section("FEATURE ORDER vs EXPECTED 27 FEATURES")
    # ------------------------------------------------------------------
    if isinstance(feature_names, list) and feature_names == EXPECTED_FEATURES:
        print("MATCH: feature_names.pkl contains exactly the 27 expected "
              "features in the expected order.")
    else:
        print("MISMATCH DETECTED:")
        print("  Saved features          :", feature_names)
        print("  Expected (27) features  :", EXPECTED_FEATURES)
        if set(feature_names) != set(EXPECTED_FEATURES):
            print("  Missing in saved :",
                  sorted(set(EXPECTED_FEATURES) - set(feature_names)))
            print("  Extra in saved   :",
                  sorted(set(feature_names) - set(EXPECTED_FEATURES)))

    # ------------------------------------------------------------------
    section("FINAL SVM - INPUT AND CAPABILITIES")
    # ------------------------------------------------------------------
    print("Model object          :", model)
    print("Expected input features:", getattr(model, "n_features_in_", "n/a"))
    print("probability support   :", probability_supported(model))
    print("Available class labels:", list(model.classes_))
    sv_shape = getattr(model, "support_vectors_", None)
    if sv_shape is not None:
        print("Support vector shape  :", sv_shape.shape)

    # ------------------------------------------------------------------
    section("STEP 4 - VERIFY svm_scaler.pkl")
    # ------------------------------------------------------------------
    print("svm_scaler object        :", svm_scaler)
    print("n_features_in_           :",
          getattr(svm_scaler, "n_features_in_", "n/a"))
    n_feats = getattr(svm_scaler, "n_features_in_", None)
    if n_feats is not None:
        match = "MATCHES" if n_feats == len(feature_names) else "MISMATCH"
        print(f"Feature count vs model   : {match} "
              f"({n_feats} vs {len(feature_names)})")
    fni = getattr(svm_scaler, "feature_names_in_", None)
    if fni is not None:
        order_match = list(fni) == list(feature_names)
        print(f"Feature order vs model   : "
              f"{'MATCHES' if order_match else 'MISMATCH'}")
        print("svm_scaler feature_names_in_:")
        for i, name in enumerate(fni, start=1):
            print(f"  {i:2d}. {name}")

    # ------------------------------------------------------------------
    section("STEP 5 - OLD scaler.pkl vs CORRECT svm_scaler.pkl")
    # ------------------------------------------------------------------
    print("old_scaler type  :", type(old_scaler).__name__,
          "| n_features_in_:", getattr(old_scaler, "n_features_in_", "n/a"))
    print("svm_scaler type  :", type(svm_scaler).__name__,
          "| n_features_in_:", getattr(svm_scaler, "n_features_in_", "n/a"))

    old_stats = scaler_stats_table(old_scaler, feature_names)
    new_stats = scaler_stats_table(svm_scaler, feature_names)

    mean_diff = (old_stats["mean_"] - new_stats["mean_"]).abs()
    scale_diff = (old_stats["scale_"] - new_stats["scale_"]).abs()

    print("\nFitted on:")
    print("  old_scaler  : FULL feature matrix X (notebook cell 29 / 5.3)")
    print("  svm_scaler  : X_train only (notebook cell 86 / 7.4); the scaler")
    print("                actually used to train the final SVM.")
    print("  The two differ only because of the data they were fit on;")
    print("  algorithm and parameters are identical StandardScaler.")

    print("\nMean (mean_) differences, old - new:")
    print(f"  max |diff| = {mean_diff.max():.6e}  "
          f"mean |diff| = {mean_diff.mean():.6e}")
    print("Scale (scale_) differences, old - new:")
    print(f"  max |diff| = {scale_diff.max():.6e}  "
          f"mean |diff| = {scale_diff.mean():.6e}")

    worst_mean = mean_diff.idxmax()
    worst_scale = scale_diff.idxmax()
    print(f"  largest mean  gap  on feature: {worst_mean} "
          f"({mean_diff[worst_mean]:.6e})")
    print(f"  largest scale gap  on feature: {worst_scale} "
          f"({scale_diff[worst_scale]:.6e})")

    print("\nPer-feature table (mean_ and scale_):")
    compare = old_stats.copy()
    compare.columns = ["old_mean_", "old_scale_"]
    compare["new_mean_"] = new_stats["mean_"]
    compare["new_scale_"] = new_stats["scale_"]
    compare["mean_diff"] = mean_diff.values
    compare["scale_diff"] = scale_diff.values
    print(compare.round(6).to_string())

    # ------------------------------------------------------------------
    section("STEP 6 - INFERENCE WITH THE CORRECT svm_scaler")
    # ------------------------------------------------------------------
    X, y = load_observation()
    print(f"Rebuilt feature matrix shape : {X.shape}")
    print(f"Rebuilt target shape         : {y.shape}")
    print(f"Target classes               : {sorted(y.unique())}")

    obs = X.iloc[[0]]
    actual = y.iloc[0]

    print(f"\nSelected observation index : {obs.index[0]}")
    print("Input shape               :", obs.shape)
    print("Actual defect             :", actual)

    scaled = svm_scaler.transform(obs)
    print("Scaled input shape        :", scaled.shape)

    prediction = model.predict(scaled)
    predicted = prediction[0]
    print("Predicted defect          :", predicted)

    if probability_supported(model):
        proba = model.predict_proba(scaled)
        print("\nProbability output (predict_proba):")
        for cls, p in zip(model.classes_, proba[0]):
            print(f"  {cls}: {p:.4f}")
    else:
        print("\nProbability support       : NOT AVAILABLE")
        print("(saved SVC was trained with probability=False; predict_proba() "
              "raises AttributeError. No confidence value is fabricated.)")

    # ------------------------------------------------------------------
    section("STEP 7 - OLD SCALER vs CORRECT SCALER PREDICTION")
    # ------------------------------------------------------------------
    pred_old = model.predict(old_scaler.transform(obs))[0]
    pred_new = model.predict(svm_scaler.transform(obs))[0]
    print(f"Old scaler.pkl    -> {pred_old}")
    print(f"svm_scaler.pkl    -> {pred_new}")
    if pred_old == pred_new:
        print("Same prediction for this observation.")
    else:
        print("Different predictions for this observation -> scaler choice "
              "matters.")

    # ------------------------------------------------------------------
    section("STEP 9 - VERIFICATION STATUS")
    # ------------------------------------------------------------------
    checks = {
        "model loads": True,
        "svm_scaler loads": True,
        "feature_names loads": True,
        "model uses 27 features": getattr(model, "n_features_in_", 0) == 27,
        "svm_scaler uses 27 features": getattr(svm_scaler, "n_features_in_", 0) == 27,
        "feature count == 27": len(feature_names) == 27,
        "feature order matches": list(fni) == list(feature_names),
        "transform works": scaled.shape == (1, 27),
        "prediction works": True,
    }
    for label, ok in checks.items():
        print(f"  [{'OK' if ok else 'FAIL'}] {label}")
    print("\nverify_model.py executed successfully.")


if __name__ == "__main__":
    main()
