# Steel Plate Quality Control Using Machine Learning

Course project: *A Data Mining Approach to Manufacturing Quality Control: Classification and Analysis of Steel Plate Defects*.

## Overview

This project develops an end-to-end machine-learning workflow for steel plate defect analysis. Using the provided steel plate fault dataset, the project covers data understanding, preprocessing, supervised classification, SVM hyperparameter tuning, unsupervised K-Means clustering, outlier detection, and Random Forest feature importance analysis.

A tuned Support Vector Machine with an RBF kernel was selected as the final classification model. Its model and preprocessing artifacts were saved, a reusable prediction module was implemented, and a Streamlit-based prediction interface was built on top of it.

The application is a machine-learning decision-support prototype for educational purposes. It is not an industrial inspection system and was not deployed or certified for production use.

## Project Objectives

- Inspect and understand the dataset.
- Prepare the data for machine-learning analysis.
- Convert the seven binary defect indicators into a single target variable.
- Train and evaluate multiple classification models.
- Establish a Linear SVM baseline.
- Tune an SVM using cross-validation.
- Compare classification models.
- Perform K-Means clustering.
- Analyze potential outliers using multiple methods.
- Analyze feature importance using Random Forest.
- Select a final classification model.
- Save the model and preprocessing artifacts.
- Develop a reusable prediction module.
- Develop a Streamlit user interface.

## Dataset

The initial dataset contains 1941 observations and 34 columns: numeric steel plate measurement/feature variables plus seven binary defect indicator columns.

The seven defect classes are:

- Pastry
- Z_Scratch
- K_Scatch
- Stains
- Dirtiness
- Bumps
- Other_Faults

The seven binary defect indicators were converted into a single categorical target variable, `Defect_Type`. After target construction and removal of the indicator columns, the dataset contains 1941 rows and 28 columns: 27 predictive features and 1 target variable.

- Missing values: 0
- Duplicate rows: 0
- Every observation has exactly one defect label.

Defect distribution:

| Defect Type | Count |
|---|---:|
| Pastry | 158 |
| Z_Scratch | 190 |
| K_Scatch | 391 |
| Stains | 72 |
| Dirtiness | 55 |
| Bumps | 402 |
| Other_Faults | 673 |

## Machine Learning Workflow

### 1. Data Understanding

The dataset was inspected for missing values (none found), duplicate rows (none found), data types, and descriptive statistics. The class distribution was found to be imbalanced, with `Other_Faults`, `Bumps`, and `K_Scatch` being the most frequent classes.

### 2. Data Preprocessing

The seven binary defect indicators were converted into the `Defect_Type` target, the original indicator columns were removed from the feature matrix, the data was split into features (`X`) and target (`y`), and a train/test split was performed. Feature scaling (standardization) was applied for the SVM models, and the feature order used during training was preserved for later prediction.

### 3. Classification

Four classification models were evaluated:

- Linear SVM
- Tuned RBF SVM
- Decision Tree
- Naive Bayes

### 4. Hyperparameter Tuning

The SVM was tuned using cross-validation with grid search:

- Best model: SVM
- Kernel: RBF
- C: 10
- Cross-validation F1: 0.7621

### 5. Model Comparison

The tuned RBF SVM achieved the best overall accuracy and F1-score among the evaluated classifiers and was selected as the final classification model.

### 6. K-Means Clustering

K-Means clustering was applied to explore natural groupings in the data, and the elbow method was used to examine inertia across `K`. K = 6 was selected.

### 7. Outlier Detection

Potential outliers were investigated using three complementary methods:

- Z-score
- IQR
- Isolation Forest

### 8. Feature Importance

A Random Forest model was used to estimate feature importance. The most important features included `LogOfAreas`, `Length_of_Conveyer`, `Pixels_Areas`, `Log_X_Index`, and `Outside_X_Index`.

### 9. Final Model

A tuned SVM with RBF kernel and C = 10 was selected as the final classification model.

## Classification Results

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Tuned SVM (C=10, RBF) | 0.7635 | 0.7672 | 0.7635 | 0.7644 |
| Decision Tree | 0.7455 | 0.7492 | 0.7455 | 0.7452 |
| Linear SVM | 0.7326 | 0.7341 | 0.7326 | 0.7326 |
| Naive Bayes | 0.4524 | 0.5741 | 0.4524 | 0.4003 |

The tuned RBF SVM was selected as the final classification model because it achieved the strongest overall classification performance on both accuracy and F1-score.

## Final Prediction Pipeline

The inference pipeline is:

```
User input
→ validation
→ feature ordering
→ svm_scaler.transform()
→ tuned SVM
→ predicted defect
```

The prediction implementation is in `src/predict.py`. It validates that the input is a DataFrame, that the exact required feature set is present, that all values are numeric, and that no values are missing or NaN, and it reorders features according to the saved feature list before prediction.

## Streamlit Application

The application is located at `app/app.py`.

Run command:

```
.venv\Scripts\python -m streamlit run app/app.py
```

The application:

- provides 27 feature inputs
- groups the inputs into logical sections
- provides an example input loaded from the dataset
- predicts the steel plate defect type
- displays model information in the sidebar
- includes a disclaimer
- uses the reusable prediction module (`src/predict.py`)
- does not display fabricated probability/confidence values

Browser testing of the application was successfully completed. Note that the SVM does not provide probability output because the saved model was trained with `probability=False`; no confidence score is shown or claimed.

## Project Structure

```
steel-plate-quality-control/
│
├── app/
│   └── app.py
│
├── data/
│   ├── Faults.NNA
│   └── Faults27x7_var
│
├── notebooks/
│   └── steel_plate_quality_control.ipynb
│
├── outputs/
│   ├── .matplotlib/
│   ├── figures/          # (empty; plots are shown inline in the notebook)
│   └── models/
│       ├── feature_names.pkl
│       ├── scaler.pkl
│       ├── svm_scaler.pkl
│       └── tuned_svm_model.pkl
│
├── reports/
│   └── STEEL_PLATE_QUALITY_CONTROL_LAB_REPORT.md
│
├── src/
│   ├── __init__.py
│   ├── extract_svm_scaler.py
│   ├── predict.py
│   ├── test_prediction.py
│   └── verify_model.py
│
├── README.md
├── REAL_APPLICATION_ROADMAP.md
└── requirements.txt
```

## Installation

Create and activate a virtual environment:

```
python -m venv .venv
```

Windows activation:

```
.venv\Scripts\activate
```

Install the dependencies:

```
pip install -r requirements.txt
```

## Running the Application

```
.venv\Scripts\python -m streamlit run app/app.py
```

The application loads the saved model and preprocessing artifacts. It does not retrain the model.

## Model Artifacts

```
outputs/models/
├── tuned_svm_model.pkl
├── svm_scaler.pkl
├── feature_names.pkl
└── scaler.pkl
```

- `tuned_svm_model.pkl` — the final trained RBF SVM (SVC, C = 10).
- `svm_scaler.pkl` — the StandardScaler fitted on the training data, used for SVM inference.
- `feature_names.pkl` — the expected 27 feature names and their order.
- `scaler.pkl` — a legacy scaler fitted on the full dataset, retained for audit/documentation purposes. It is **not** used for inference.

### Important scaler note

There are two scaler files:

- `outputs/models/scaler.pkl` — legacy scaler fitted on the full dataset. It is retained for audit/documentation purposes and is **not** used for final model inference.
- `outputs/models/svm_scaler.pkl` — the correct scaler for the final SVM. It was fitted on the training data and is used by `src/predict.py`.

The final inference pipeline uses `outputs/models/svm_scaler.pkl`.

## Limitations

- The final model achieves approximately 76% classification accuracy, so predictions are not always correct.
- The dataset has a class imbalance, and no balancing technique was applied.
- The outlier methods identify potential statistical anomalies, not guaranteed data errors or physically defective plates.
- The Streamlit application is a prototype/decision-support tool, not a certified industrial inspection system.
- The current SVM does not provide probability/confidence output because the saved model was trained with `probability=False`.
- The results are based on the provided dataset and should not automatically be generalized to other steel manufacturing environments.

## Lab Report

The detailed academic analysis is documented in:

`reports/STEEL_PLATE_QUALITY_CONTROL_LAB_REPORT.md`

The lab report covers the data analysis, preprocessing, model experiments, evaluation, clustering, outlier analysis, feature importance, final model selection, and application documentation in detail.

## Roadmap

`REAL_APPLICATION_ROADMAP.md` contains the original project/application planning roadmap. Some filenames in the roadmap are aspirational examples from the planning stage and differ from the final implementation.

## License / Academic Note

This repository represents an academic/course project demonstrating a machine-learning workflow. It is provided for educational purposes and is not licensed for commercial or production use.
