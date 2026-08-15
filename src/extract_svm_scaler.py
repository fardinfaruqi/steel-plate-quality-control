"""Save the SVM-specific scaler used by the notebook for the final SVM.

Background
----------
The notebook creates TWO StandardScaler objects:

  * ``scaler``     (section 5.3, cell 29):  fitted on the FULL feature
                  matrix ``X``. This object was mistakenly saved as
                  ``outputs/models/scaler.pkl`` (cell 320).

  * ``svm_scaler`` (section 7.4, cell 86):  fitted on ``X_train`` ONLY
                  and used to produce ``X_train_scaled`` / ``X_test_scaled``
                  for the SVM workflow (cells 88, 125, 133). The final
                  tuned SVM (``best_svm``, cell 131) was trained on
                  ``X_train_scaled``, so ``svm_scaler`` is the correct
                  preprocessing artifact for inference.

Because this script cannot access the notebook's live kernel, it
reproduces ``svm_scaler`` by re-running exactly the deterministic steps
of the notebook that create it:

    load data (cell 3)
      -> build Defect_Type target (cell 21)
      -> X, y split (cell 25)
      -> train_test_split(test_size=0.20, random_state=42, stratify=y)
           (cell 57)
      -> svm_scaler.fit(X_train) (cell 86)

``train_test_split`` and ``StandardScaler`` are deterministic, so the
reproduced scaler statistics are identical to those held in the
notebook kernel when the model was saved.

This script does NOT train any classifier and does NOT overwrite the
existing ``scaler.pkl``, ``tuned_svm_model.pkl``, or ``feature_names.pkl``.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

DATA_PATH = DATA_DIR / "Faults.NNA"
COLUMNS_PATH = DATA_DIR / "Faults27x7_var"
OUT_PATH = MODELS_DIR / "svm_scaler.pkl"

FAULT_COLUMNS = [
    "Pastry",
    "Z_Scratch",
    "K_Scatch",
    "Stains",
    "Dirtiness",
    "Bumps",
    "Other_Faults",
]


def main():
    with open(COLUMNS_PATH, "r") as f:
        columns = [line.strip() for line in f]

    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=columns)

    # Notebook cell 21: build target, drop the 7 fault columns.
    df["Defect_Type"] = df[FAULT_COLUMNS].idxmax(axis=1)
    df = df.drop(columns=FAULT_COLUMNS)

    # Notebook cell 25: separate features and target.
    X = df.drop(columns=["Defect_Type"])
    y = df["Defect_Type"]

    # Notebook cell 57: train/test split (deterministic).
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Notebook cell 86: svm_scaler fitted ONLY on X_train.
    svm_scaler = StandardScaler()
    svm_scaler.fit(X_train)

    joblib.dump(svm_scaler, OUT_PATH)

    print(f"Saved svm_scaler.pkl -> {OUT_PATH}")
    print(f"Scaler type          : {type(svm_scaler).__name__}")
    print(f"n_features_in_       : {svm_scaler.n_features_in_}")
    print(f"Fitted on X_train    : {X_train.shape}")
    print("feature_names_in_ order:")
    for i, name in enumerate(svm_scaler.feature_names_in_, start=1):
        print(f"  {i:2d}. {name}")


if __name__ == "__main__":
    main()
