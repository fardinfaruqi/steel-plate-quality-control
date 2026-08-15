"""Reusable prediction module for the Steel Plate Quality Control project.

Loads the saved production artifacts and turns 27 steel-plate feature
measurements into a predicted defect type.

Artifacts used (DO NOT change these):
  outputs/models/tuned_svm_model.pkl  - final tuned SVM (SVC C=10, RBF)
  outputs/models/svm_scaler.pkl       - scaler fitted on X_train (the scaler
                                         used to train the final SVM)
  outputs/models/feature_names.pkl    - the 27 features in model order

The old outputs/models/scaler.pkl (fitted on the FULL dataset) is NOT used.
The scaler is never refitted during prediction.

Typical usage:
    from predict import predict_defect, predict_defects

    prediction = predict_defect(one_row_df)       # -> defect string
    batch      = predict_defects(many_rows_df)    # -> pd.Series of defects
"""

from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

MODEL_PATH = MODELS_DIR / "tuned_svm_model.pkl"
SCALER_PATH = MODELS_DIR / "svm_scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"

_artifacts = None


def load_artifacts():
    """Load (once) and return ``(model, scaler, feature_names)``.

    Artifacts are loaded lazily and cached so repeated calls do not
    re-read the files. ``feature_names`` is the list of the 27 required
    features in the exact order the SVM was trained on.
    """
    global _artifacts
    if _artifacts is None:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        _artifacts = (model, scaler, feature_names)
    return _artifacts


def get_feature_names():
    """Return the list of the 27 required features (model order)."""
    _, _, feature_names = load_artifacts()
    return list(feature_names)


def _prepare_input(input_data):
    """Validate ``input_data`` and return it reordered to model order.

    Checks, in order:
      1. Input must be a pandas DataFrame.
      2. It must contain exactly the required features (no missing,
         no extra).
      3. All required columns must be numeric.
      4. No required value may be missing (NaN).

    The returned DataFrame always has its columns in the exact order of
    feature_names.pkl, regardless of the order the caller supplied.

    Raises
    ------
    TypeError
        If ``input_data`` is not a pandas DataFrame.
    ValueError
        If features are missing/extra, non-numeric, or contain NaN.
    """
    model, _, feature_names = load_artifacts()

    if not isinstance(input_data, pd.DataFrame):
        raise TypeError(
            "input_data must be a pandas DataFrame with the 27 feature "
            f"columns, got {type(input_data).__name__}."
        )

    missing = [f for f in feature_names if f not in input_data.columns]
    extra = [c for c in input_data.columns if c not in feature_names]

    if missing:
        raise ValueError(
            "Missing required feature column(s): "
            f"{missing}. Expected columns: {feature_names}."
        )
    if extra:
        raise ValueError(
            "Unexpected extra column(s) not used by the model: "
            f"{extra}. Expected exactly the 27 features: {feature_names}."
        )

    ordered = input_data[feature_names]

    non_numeric = [
        c for c in feature_names
        if not pd.api.types.is_numeric_dtype(ordered[c])
    ]
    if non_numeric:
        raise ValueError(
            "Feature column(s) contain non-numeric values: "
            f"{non_numeric}. All features must be numeric."
        )

    ordered = ordered.astype(float)

    if ordered.isna().any().any():
        raise ValueError(
            "Feature value(s) are missing (NaN) in the input. "
            "Every required feature must have a numeric value."
        )

    return ordered


def predict_defect(input_data):
    """Predict the defect type for exactly ONE steel plate.

    Parameters
    ----------
    input_data : pandas.DataFrame
        A single-row DataFrame with the 27 required feature columns.
        Feature order does not matter; columns are reordered internally
        using feature_names.pkl.

    Returns
    -------
    str
        The predicted defect class (e.g. ``"K_Scatch"``).

    Raises
    ------
    TypeError
        If ``input_data`` is not a pandas DataFrame.
    ValueError
        If it does not contain exactly one row, if features are
        missing/extra, non-numeric, or contain NaN.

    Preprocessing
    -------------
    Columns are reordered to model order, then scaled with the saved
    ``svm_scaler`` (fitted on X_train during training). The scaler is
    never refitted. No probability is returned because the saved SVC
    was trained with ``probability=False``.
    """
    model, scaler, _ = load_artifacts()

    if len(input_data) != 1:
        raise ValueError(
            "predict_defect expects exactly one observation (one row). "
            f"Got {len(input_data)} rows. Use predict_defects() for "
            "multiple observations."
        )

    ordered = _prepare_input(input_data)
    scaled = scaler.transform(ordered)
    prediction = model.predict(scaled)
    return prediction[0]


def predict_defects(input_data):
    """Predict the defect type for one or more steel plates (batch).

    Parameters
    ----------
    input_data : pandas.DataFrame
        A DataFrame with the 27 required feature columns and any number
        of rows. Feature order does not matter.

    Returns
    -------
    pandas.Series
        Predicted defect class per row, named ``"Predicted_Defect"``
        and indexed like ``input_data``.

    Raises
    ------
    TypeError
        If ``input_data`` is not a pandas DataFrame.
    ValueError
        If it contains zero rows, if features are missing/extra,
        non-numeric, or contain NaN.

    Preprocessing
    -------------
    Same as ``predict_defect``: reorder to model order, scale with the
    saved ``svm_scaler``, predict. No probability is returned.
    """
    model, scaler, _ = load_artifacts()

    if len(input_data) == 0:
        raise ValueError(
            "predict_defects expects at least one observation "
            "(one row). Got an empty DataFrame."
        )

    ordered = _prepare_input(input_data)
    scaled = scaler.transform(ordered)
    predictions = model.predict(scaled)
    return pd.Series(predictions, index=input_data.index, name="Predicted_Defect")
