"""Steel Plate Defect Predictor - Streamlit user interface.

This UI is a thin layer over the reusable prediction module
(src/predict.py). All model loading, feature ordering, scaling and
prediction happen inside src/predict.py; this file only collects the
user's inputs and displays the result.

Run from the project root with:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from src.predict import predict_defect, get_feature_names

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "Faults.NNA"
COLUMNS_PATH = DATA_DIR / "Faults27x7_var"

FAULT_COLUMNS = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains",
    "Dirtiness", "Bumps", "Other_Faults",
]

# Binary (0/1) features -> shown as a small selectbox instead of a spinner.
BINARY_FEATURES = {"TypeOfSteel_A300", "TypeOfSteel_A400"}

# Integer-valued features (checked against the raw dataset dtypes) ->
# integer number inputs. Everything else is decimal.
INTEGER_FEATURES = {
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity",
    "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Length_of_Conveyer",
    "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness",
}

# UI-only grouping. It never changes feature names or the order of the
# values passed to predict_defect (which reorders internally).
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

BINARY_HELP = {
    "TypeOfSteel_A300": "1 if the plate is steel type A300, otherwise 0.",
    "TypeOfSteel_A400": "1 if the plate is steel type A400, otherwise 0.",
}

MODEL_INFO = {
    "Final Model": "RBF SVM",
    "C": "10",
    "Number of Features": "27",
    "Test Accuracy": "76.35%",
    "Test F1-Score": "76.44%",
}

DISCLAIMER = (
    "This application provides a machine-learning prediction based on the "
    "trained dataset. It should be considered a decision-support tool and "
    "should not replace physical inspection or established industrial "
    "quality-control procedures."
)


@st.cache_data
def load_examples():
    """Return the notebook's X (27 features) and y (Defect_Type) for the
    optional 'Try an Example' section. The target column is used only to
    label examples for reference; it is never used as model input."""
    with open(COLUMNS_PATH, "r") as f:
        columns = [line.strip() for line in f]
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=columns)
    df["Defect_Type"] = df[FAULT_COLUMNS].idxmax(axis=1)
    df = df.drop(columns=FAULT_COLUMNS)
    X = df.drop(columns=["Defect_Type"])
    y = df["Defect_Type"]
    return X, y


def widget_key(feature):
    return f"input_{feature}"


def render_feature_input(feature):
    """Render the Streamlit widget for one feature, keyed by feature name.

    Defaults are managed through st.session_state so the example loader
    can populate the form without tripping Streamlit's widget policy
    (no default ``value``/``index`` is passed alongside a session-state
    value).
    """
    key = widget_key(feature)

    if key not in st.session_state:
        if feature in INTEGER_FEATURES:
            st.session_state[key] = 0
        elif feature in BINARY_FEATURES:
            st.session_state[key] = 0
        else:
            st.session_state[key] = 0.0

    if feature in BINARY_FEATURES:
        return st.selectbox(
            feature,
            options=[0, 1],
            format_func=lambda v: "1" if v == 1 else "0",
            key=key,
            help=BINARY_HELP[feature],
        )

    if feature in INTEGER_FEATURES:
        return st.number_input(feature, step=1, key=key)

    return st.number_input(feature, step=0.001, format="%.4f", key=key)


def main():
    st.set_page_config(
        page_title="Steel Plate Defect Predictor",
        page_icon="\U0001F3F7",
        layout="wide",
    )

    feature_names = get_feature_names()

    # ------------------------------------------------------------------ #
    # Sidebar: model information
    # ------------------------------------------------------------------ #
    with st.sidebar:
        st.markdown("### Model Information")
        for label, value in MODEL_INFO.items():
            st.markdown(f"**{label}:** {value}")
        st.markdown(
            "No probability/confidence score is displayed because the "
            "trained SVC does not support probability prediction."
        )
        st.divider()
        st.caption(
            f"Model: tuned SVM on {len(feature_names)} features "
            "(SVC, C=10, RBF kernel)."
        )

    # ------------------------------------------------------------------ #
    # Header
    # ------------------------------------------------------------------ #
    st.title("Steel Plate Defect Predictor")
    st.markdown(
        "This application uses a trained Support Vector Machine (SVM) model "
        "to predict the type of defect detected on a steel plate based on "
        "27 measured features."
    )

    with st.expander("About the model"):
        st.markdown(
            f"- **Model:** Tuned Support Vector Machine\n"
            f"- **Kernel:** RBF\n"
            f"- **C:** 10\n"
            f"- **Number of input features:** 27\n"
            f"- **Test Accuracy:** 76.35%\n"
            f"- **Test F1-Score:** 76.44%"
        )

    # ------------------------------------------------------------------ #
    # Optional: Try an Example
    # ------------------------------------------------------------------ #
    with st.expander("Try an Example"):
        st.caption(
            "Load a real observation from the original dataset into the "
            "form. The actual defect shown below is only for reference; it "
            "is never used as model input. The model may not always "
            "predict the example correctly."
        )
        X_ex, y_ex = load_examples()
        example_idx = st.selectbox(
            "Choose an example (dataset row)",
            options=list(range(min(10, len(X_ex)))),
            format_func=lambda i: (
                f"Row {i} - actual defect in dataset: {y_ex.iloc[i]}"
            ),
        )
        if st.button("Load Example Values", use_container_width=True):
            for feature in feature_names:
                value = X_ex.iloc[example_idx][feature]
                if feature in INTEGER_FEATURES:
                    value = int(value)
                else:
                    value = float(value)
                st.session_state[widget_key(feature)] = value
            st.success(
                f"Example loaded from row {example_idx}. "
                "Click 'Predict Defect' below to run the model."
            )

    st.divider()

    # ------------------------------------------------------------------ #
    # Input form (grouped)
    # ------------------------------------------------------------------ #
    st.markdown("### Plate Measurements")
    st.caption(
        "Enter the 27 measured features. Feature names follow the trained "
        "model; values are reordered internally to the exact model order."
    )

    for group_name, group_features in FEATURE_GROUPS:
        st.markdown(f"#### {group_name}")
        columns = st.columns(3)
        for i, feature in enumerate(group_features):
            with columns[i % 3]:
                render_feature_input(feature)
        st.write("")

    st.divider()

    # ------------------------------------------------------------------ #
    # Prediction
    # ------------------------------------------------------------------ #
    if st.button("Predict Defect", type="primary", use_container_width=True):
        try:
            row = {
                feature: st.session_state[widget_key(feature)]
                for feature in feature_names
            }
            input_df = pd.DataFrame([row])

            prediction = predict_defect(input_df)

            st.markdown("### Predicted Defect")
            st.markdown(f"## {prediction}")
            st.success("Prediction completed using the tuned SVM model.")
        except (TypeError, ValueError) as exc:
            st.error(f"Invalid input: {exc}")
        except Exception as exc:  # noqa: BLE001 - surface unexpected failures
            st.error(f"Prediction failed: {exc}")

    st.divider()
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
