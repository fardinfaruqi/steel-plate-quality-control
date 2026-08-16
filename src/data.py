"""Shared dataset helpers for the Steel Plate Quality Control project.

The raw data is stored as two files: ``Faults.NNA`` (tab-separated, no
header) and ``Faults27x7_var`` (the 34 column names, one per line). This
module centralizes the loading, target construction, and train/test split
so every consumer (prediction module, verification scripts, API) uses the
exact same data preparation as the notebook.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "Faults.NNA"
COLUMNS_PATH = DATA_DIR / "Faults27x7_var"

FAULT_COLUMNS = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains",
    "Dirtiness", "Bumps", "Other_Faults",
]

# Mirrors the notebook's Section 7.1 split (stratified, deterministic).
SPLIT_PARAMS = {
    "test_size": 0.20,
    "random_state": 42,
}


def load_plate_data():
    """Return (X, y) rebuilt exactly as the notebook does.

    X holds the 27 predictive features in dataset column order; y holds
    the ``Defect_Type`` target derived from the 7 binary fault indicators.
    """
    with open(COLUMNS_PATH, "r") as f:
        columns = [line.strip() for line in f]

    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=columns)
    df["Defect_Type"] = df[FAULT_COLUMNS].idxmax(axis=1)
    df = df.drop(columns=FAULT_COLUMNS)

    X = df.drop(columns=["Defect_Type"])
    y = df["Defect_Type"]
    return X, y


def train_test_split_plates(X, y):
    """Reproduce the notebook's exact train/test split.

    Uses ``test_size=0.20`` and ``random_state=42`` with stratification,
    identical to the notebook's Section 7.1 cell.
    """
    return train_test_split(
        X,
        y,
        test_size=SPLIT_PARAMS["test_size"],
        random_state=SPLIT_PARAMS["random_state"],
        stratify=y,
    )
