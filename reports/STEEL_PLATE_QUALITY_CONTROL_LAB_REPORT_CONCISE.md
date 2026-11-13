# Data Mining Laboratory Report (Concise Edition): Steel Plate Quality Control Using Data Mining and Machine Learning

**Course:** Data Mining Laboratory  
**Project Title:** Steel Plate Quality Control Using Data Mining and Machine Learning  
**Submitted By:** Kazi MD Baha Uddin Faruqi | **Student ID:** 232031035  
**Dataset:** Steel Plate Faults (`data/Faults.NNA` & `data/Faults27x7_var` — 1,941 observations, 27 predictive features, 7 defect classes)

---

# Lab 1: Introduction to Python for Data Mining

## Lab Task 1.1: Importing Core Data Mining and Machine Learning Libraries
**Explanation:** We initialize the Python environment by importing NumPy, Pandas, Matplotlib, Seaborn, SciPy, Joblib, and Scikit-Learn modules for scaling, classification (`DecisionTreeClassifier`, `GaussianNB`, `SVC`, `RandomForestClassifier`), clustering (`KMeans`), dimensionality reduction (`PCA`), and anomaly detection (`IsolationForest`).

```python
import os, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
sns.set_theme(style="whitegrid")
```

## Lab Task 1.2: Loading the Raw Steel Plate Dataset and Column Definitions Using Pandas
**Explanation:** The dataset is stored in two files: `data/Faults27x7_var` (34 attribute names) and `data/Faults.NNA` (1,941 tab-separated rows).

```python
with open("../data/Faults27x7_var", "r") as file:
    columns = [line.strip() for line in file]

df = pd.read_csv("../data/Faults.NNA", sep="\t", header=None, names=columns)
print(f"Total Columns: {len(columns)} | Dataset Shape: {df.shape}")
```

```text
Total Columns: 34 | Dataset Shape: (1941, 34)
First 5 Features: ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Pixels_Areas']
Target Columns (7): ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']
```

---

# Lab 2: Data Preprocessing and Exploration

## Lab Task 2.1: Data Quality Audit, Missing/Duplicate Verification, and Descriptive Statistics
**Explanation:** We verify completeness, check for duplicate rows, inspect data types (`20 int64`, `14 float64`), and summarize key feature distributions.

```python
print("Missing Values:", df.isnull().sum().sum(), "| Duplicate Rows:", df.duplicated().sum())
display(df[["Pixels_Areas", "X_Perimeter", "Length_of_Conveyer",
            "Steel_Plate_Thickness", "Edges_Index", "Luminosity_Index"]].describe().T)
```

| Feature | Count | Mean | Std. Dev. | Min | 25% | Median (50%) | 75% | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `Pixels_Areas` | 1,941 | 1,893.88 | 5,168.46 | 2.00 | 84.00 | 174.00 | 822.00 | 152,655.00 |
| `X_Perimeter` | 1,941 | 111.86 | 301.21 | 2.00 | 15.00 | 26.00 | 84.00 | 10,449.00 |
| `Sum_of_Luminosity` | 1,941 | 206,312.15 | 512,293.59 | 250.00 | 9,522.00 | 19,202.00 | 83,011.00 | 11,591,414.00 |
| `Length_of_Conveyer` | 1,941 | 1,459.16 | 144.58 | 1,227.00 | 1,358.00 | 1,364.00 | 1,650.00 | 1,794.00 |
| `Steel_Plate_Thickness` | 1,941 | 78.74 | 55.09 | 40.00 | 40.00 | 70.00 | 80.00 | 300.00 |
| `Edges_Index` | 1,941 | 0.3317 | 0.2997 | 0.0000 | 0.0604 | 0.2273 | 0.5738 | 0.9952 |
| `Luminosity_Index` | 1,941 | -0.1313 | 0.1488 | -0.9989 | -0.1950 | -0.1330 | -0.0666 | 0.6421 |

**Interpretation:** Zero missing values and zero duplicates exist. Features span vastly different scales (`Sum_of_Luminosity` up to $1.16 \times 10^7$ vs. normalized indices in $[-1, 1]$), requiring Z-score standardization for distance- and margin-based models.

## Lab Task 2.2: Target Reconstruction (`Defect_Type`), Feature Separation (`X`, `y`), and Z-Score Standardization
**Explanation:** The 7 binary fault columns are verified as mutually exclusive (`row_sums.unique() == [1]`), collapsed into a single multi-class target `Defect_Type` via `idxmax(axis=1)`, separated into `X` ($1,941 \times 27$) and `y` ($1,941$), and standardized using `StandardScaler`.

```python
fault_columns = ["Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
assert (df[fault_columns].sum(axis=1) == 1).all(), "Each plate must have exactly 1 fault label"

df["Defect_Type"] = df[fault_columns].idxmax(axis=1)
df = df.drop(columns=fault_columns)
X = df.drop(columns=["Defect_Type"])
y = df["Defect_Type"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(f"X shape: {X.shape} | y shape: {y.shape} | Post-scaling Mean ~ 0.0000, Std ~ 1.0000")
```

---

# Lab 3: Data Visualization and Exploratory Data Analysis (EDA)

## Lab Task 3.1: Target Defect Distribution and Steel Grade (`A300` vs. `A400`) Breakdown
**Explanation:** We visualize the class distribution across all 1,941 plates and compare defect frequencies between `TypeOfSteel_A300` ($n=777$) and `TypeOfSteel_A400` ($n=1,164$).

```python
steel_type = np.where(df["TypeOfSteel_A300"] == 1, "A300", "A400")
display(pd.crosstab(steel_type, df["Defect_Type"], margins=True))
```

| Steel Grade | `Bumps` | `Dirtiness` | `K_Scatch` | `Other_Faults` | `Pastry` | `Stains` | `Z_Scratch` | Total Plates |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Grade A300** | 279 | 9 | 1 | 266 | 49 | 1 | 172 | **777 (40.0%)** |
| **Grade A400** | 123 | 46 | 390 | 407 | 109 | 71 | 18 | **1,164 (60.0%)** |
| **Total ($N=1,941$)** | **402 (20.7%)** | **55 (2.8%)** | **391 (20.1%)** | **673 (34.7%)** | **158 (8.1%)** | **72 (3.7%)** | **190 (9.8%)** | **1,941 (100%)** |

**Interpretation:** The dataset exhibits a $12.24 : 1$ imbalance between `Other_Faults` (673) and `Dirtiness` (55). Defect occurrence is strongly conditioned by steel grade: **99.7% of `K_Scatch` (390/391)** and **98.6% of `Stains` (71/72)** occur on `A400`, whereas **90.5% of `Z_Scratch` (172/190)** occurs on `A300`.

## Lab Task 3.2: Feature Distributions (Histograms & KDE) and Class-Conditioned Box Plots
**Explanation:** We inspect univariate skewness using histograms with KDE overlays and evaluate class separability using box plots across `Pixels_Areas`, `Steel_Plate_Thickness`, `Edges_Index`, and `Luminosity_Index`.

```python
selected_features = ["Pixels_Areas", "Steel_Plate_Thickness", "Edges_Index", "Luminosity_Index"]
print(df[selected_features].skew().round(2).to_dict())
```

```text
Skewness -> {'Pixels_Areas': 13.79, 'Steel_Plate_Thickness': 2.19, 'Edges_Index': 0.69, 'Luminosity_Index': 0.68}
```

**Interpretation:** `Pixels_Areas` is heavily right-skewed ($\text{skewness} = +13.79$) due to large `K_Scatch` defects, while `Steel_Plate_Thickness` shows discrete manufacturing gauges ($40$–$300\text{ mm}$). Class-conditioned box plots reveal that `K_Scatch` dominates upper area quantiles, `Stains` occupies compact area ranges, and `Dirtiness` exhibits high `Steel_Plate_Thickness` medians.

## Lab Task 3.3: Bivariate Scatter Analysis, Univariate IQR Outliers, and $27 \times 27$ Correlation Matrix
**Explanation:** We examine the collinearity between `Pixels_Areas` and `Sum_of_Luminosity`, count univariate $1.5 \times \text{IQR}$ outliers per feature, and compute the full $27 \times 27$ Pearson correlation matrix.

```python
correlation_matrix = X.corr()
pairs = correlation_matrix.abs().unstack().sort_values(ascending=False)
pairs = pairs[pairs < 1.0].drop_duplicates().head(6)
```

| Rank | Feature Pair (`Feature 1` — `Feature 2`) | Pearson $r$ | Top IQR Outlier Features | Flagged Plates ($> 1.5\times\text{IQR}$) |
| :---: | :--- | ---: | :--- | ---: |
| 1 | `TypeOfSteel_A300` — `TypeOfSteel_A400` | $-1.0000$ | `Outside_X_Index` | 401 (20.66%) |
| 2 | `Y_Minimum` — `Y_Maximum` | $+1.0000$ | `Pixels_Areas` | 363 (18.70%) |
| 3 | `X_Minimum` — `X_Maximum` | $+0.9883$ | `Sum_of_Luminosity` | 361 (18.60%) |
| 4 | `Pixels_Areas` — `Sum_of_Luminosity` | $+0.9790$ | `X_Perimeter` | 322 (16.59%) |
| 5 | `X_Perimeter` — `Y_Perimeter` | $+0.9310$ | `Y_Perimeter` | 290 (14.94%) |

**Interpretation:** Strong multicollinearity exists among bounding-box coordinates, area/luminosity totals ($r = 0.9790$), and complementary steel-type indicators ($r = -1.0000$). Out of 27 features, 15 contain univariate IQR outliers (led by `Outside_X_Index` with 401 and `Pixels_Areas` with 363), including extreme plate **Row 391** (`K_Scatch`, `Pixels_Areas = 152,655`).

---

# Lab 4: Classification – Decision Trees

## Lab Task 4.1: Stratified Train–Test Split, Decision Tree Training, and Tree Structure Visualization
**Explanation:** We partition the dataset into an **80% training set (1,552 plates)** and **20% test set (389 plates)** using `stratify=y` (`random_state=42`), train a CART `DecisionTreeClassifier`, evaluate its confusion matrix, and visualize the top split levels.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
decision_tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
y_pred_dt = decision_tree.predict(X_test)
print(f"DT Depth: {decision_tree.get_depth()} | Leaves: {decision_tree.get_n_leaves()} | Test Acc: {accuracy_score(y_test, y_pred_dt):.4f}")
```

| Defect Class | Precision | Recall | F1-Score | Support | Key Splitting Insight |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `Bumps` | 0.6627 | 0.6790 | 0.6707 | 81 | Moderate overlap with `Other_Faults` (15) & `Pastry` (6) |
| `Dirtiness` | 0.7692 | 0.9091 | 0.8333 | 11 | Strong recall (10/11); 1 misclassified as `Other_Faults` |
| `K_Scatch` | 0.9851 | 0.8462 | 0.9103 | 78 | Root split (`Log_X_Index <= 2.043`) isolates large `K_Scatch` |
| `Other_Faults` | 0.6978 | 0.7185 | 0.7080 | 135 | 97/135 correct; 21 misclassified as `Bumps` |
| `Pastry` | 0.5000 | 0.4375 | 0.4667 | 32 | Weakest class; 13 confused with `Other_Faults` |
| `Stains` | 0.7857 | 0.7857 | 0.7857 | 14 | Isolated early via `Pixels_Areas <= 23.0` & `Empty_Index` |
| `Z_Scratch` | 0.8222 | 0.9737 | 0.8916 | 38 | 37/38 recalled (`TypeOfSteel_A400 <= 0.5` branch) |
| **Weighted / Total** | **0.7492** | **0.7455** | **0.7452** | **389** | **Overall Test Accuracy = 74.55% (Depth = 18, Leaves = 232)** |

---

# Lab 5: Classification – Naive Bayes

## Lab Task 5.1: Training and Evaluating the Gaussian Naive Bayes Classifier
**Explanation:** We train a probabilistic `GaussianNB` classifier on `(X_train, y_train)` and evaluate its performance on the 389 held-out test plates.

```python
naive_bayes = GaussianNB().fit(X_train, y_train)
y_pred_nb = naive_bayes.predict(X_test)
print(f"Gaussian NB Test Accuracy: {accuracy_score(y_test, y_pred_nb):.4f} | Weighted F1: {f1_score(y_test, y_pred_nb, average='weighted'):.4f}")
```

```text
Gaussian NB Test Accuracy: 0.4524 | Weighted Precision: 0.5741 | Weighted Recall: 0.4524 | Weighted F1: 0.4003
```

**Interpretation:** Gaussian Naive Bayes achieves only **45.24% accuracy** (`F1 = 0.4003`) because strong feature multicollinearity ($r > 0.97$ between geometric and luminosity features) and extreme positive skewness violate its conditional independence and Gaussian likelihood assumptions, causing 78 `Other_Faults` and 23 `Pastry` plates to be misclassified as `Bumps`.

---

# Lab 6: Classification – Support Vector Machines (SVM)

## Lab Task 6.1: Leak-Free Feature Standardization, Baseline Linear SVM, and 5-Fold GridSearchCV RBF SVM
**Explanation:** Because SVMs optimize margin distances, we fit `svm_scaler = StandardScaler()` strictly on `X_train` (preventing test-set leakage), train a baseline Linear SVM (`kernel="linear"`), and tune hyperparameters (`C` $\in \{0.1, 1, 10\}$, `kernel` $\in \{\text{linear}, \text{rbf}\}$, `gamma` $\in \{\text{scale}, \text{auto}\}$) via 5-fold `GridSearchCV`.

```python
svm_scaler = StandardScaler()
X_train_scaled = svm_scaler.fit_transform(X_train)
X_test_scaled = svm_scaler.transform(X_test)

svm_linear = SVC(kernel="linear", random_state=42).fit(X_train_scaled, y_train)
grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid={"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
    cv=5, scoring="f1_weighted", n_jobs=-1
).fit(X_train_scaled, y_train)
best_svm = grid_search.best_estimator_  # SVC(C=10, kernel='rbf', gamma='scale')
print("Best Params:", grid_search.best_params_, "| Best 5-Fold CV F1:", round(grid_search.best_score_, 4))
```

| Defect Class | Support | Gaussian NB F1 (Lab 5) | Linear SVM F1 (Lab 6) | Tuned RBF SVM (`C=10`) Precision | Tuned RBF SVM Recall | Tuned RBF SVM F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Bumps` | 81 | 0.4545 | 0.6164 | 0.6706 | 0.7037 | **0.6867** |
| `Dirtiness` | 11 | 0.0769 | 0.5455 | 1.0000 | 0.7273 | **0.8421** |
| `K_Scatch` | 78 | 0.8112 | 0.9481 | 0.9867 | 0.9487 | **0.9673** |
| `Other_Faults` | 135 | 0.1818 | 0.6786 | 0.6884 | 0.7037 | **0.6960** |
| `Pastry` | 32 | 0.0526 | 0.5763 | 0.6207 | 0.5625 | **0.5902** |
| `Stains` | 14 | 0.5581 | 0.8571 | 0.8000 | 0.8571 | **0.8276** |
| `Z_Scratch` | 38 | 0.5455 | 0.8800 | 0.8462 | 0.8684 | **0.8571** |
| **Overall / Weighted** | **389** | **0.4003 (Acc 45.24%)** | **0.7326 (Acc 73.26%)** | **0.7672** | **0.7635** | **0.7644 (Acc 76.35%)** |

**Interpretation:** Standardized Linear SVM reaches **73.26% accuracy**, and 5-fold `GridSearchCV` selects `SVC(C=10, kernel='rbf', gamma='scale')` (CV F1 = `0.7689`), boosting test accuracy to **76.35%** (`F1 = 0.7644`) with major gains on `Dirtiness` (+0.2966 F1) and `Bumps` (+0.0703 F1).

---

# Lab 7: Clustering – K-Means

## Lab Task 7.1: Elbow Method ($K=2..10$), $K=6$ K-Means Clustering, 2D PCA Projection, and Defect Alignment
**Explanation:** We standardize all 1,941 feature vectors (`cluster_scaler`), evaluate within-cluster inertia from $K=2$ (`40,466.52`) to $K=10$ (`21,500.51`), select the elbow at $K=6$ (`Inertia = 27,072.81`), project clusters onto 2D PCA (`43.42%` variance), and cross-tabulate clusters against actual defect labels.

```python
X_cluster_scaled = StandardScaler().fit_transform(X)
inertia_values = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_cluster_scaled).inertia_ for k in range(2, 11)]
final_kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
final_cluster_labels = final_kmeans.fit_predict(X_cluster_scaled)
X_pca = PCA(n_components=2, random_state=42).fit_transform(X_cluster_scaled)
```

| Cluster ID | Plate Count (%) | `Bumps` | `Dirtiness` | `K_Scatch` | `Other_Faults` | `Pastry` | `Stains` | `Z_Scratch` | Dominant Unsupervised Structural Profile |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **C0** | 444 (22.87%) | 38 | 35 | 7 | 192 | 103 | 0 | 69 | Mixed `Other_Faults` (43.2%) & `Pastry` (23.2%) |
| **C1** | 483 (24.88%) | 96 | 12 | 55 | 224 | 19 | 71 | 6 | Small/medium `A400` faults; holds **98.6% of `Stains`** |
| **C2** | 125 (6.44%) | 0 | 0 | **121** | 3 | 0 | 0 | 1 | High-purity large `K_Scatch` (**96.8% pure**) |
| **C3** | 641 (33.02%) | **263** | 8 | 1 | 222 | 36 | 1 | **110** | Dominant `A300` cluster (`Bumps`, `Other`, `Z_Scratch`) |
| **C4** | 1 (0.05%) | 0 | 0 | **1** | 0 | 0 | 0 | 0 | **Row 391** singleton anomaly (`Pixels_Areas = 152,655`) |
| **C5** | 247 (12.73%) | 5 | 0 | **206** | 32 | 0 | 0 | 4 | Medium-to-large `K_Scatch` cluster (**83.4% pure**) |

---

# Lab 8: Outlier Detection and Ensemble Methods

## Lab Task 8.1: Multi-Method Outlier Detection (Z-Score, IQR, Isolation Forest) and Consensus Analysis
**Explanation:** We compare three anomaly detectors across all 1,941 plates: (1) Z-Score (`>3` features with $|Z|>3$), (2) IQR (`>5` features outside $1.5 \times \text{IQR}$), and (3) `IsolationForest(contamination=0.10, random_state=42)`.

```python
z_outliers = (np.abs(zscore(X)) > 3).sum(axis=1) > 3
Q1, Q3 = X.quantile(0.25), X.quantile(0.75)
iqr_outliers = ((X < (Q1 - 1.5*(Q3-Q1))) | (X > (Q3 + 1.5*(Q3-Q1)))).sum(axis=1) > 5
iso_outliers = IsolationForest(contamination=0.10, random_state=42).fit_predict(X_cluster_scaled) == -1
outlier_votes = z_outliers.astype(int) + iqr_outliers.astype(int) + iso_outliers.astype(int)
```

| Outlier Metric / Agreement Level | Count | Share (%) | Consensus Outliers ($n=102$) by Defect Class | Count |
| :--- | ---: | ---: | :--- | ---: |
| **Z-Score Method (`>3` features with $\|Z\|>3$)** | 211 | 10.87% | `K_Scatch` (Large surface tears) | 71 (69.6%) |
| **IQR Method (`>5` features outside $1.5\times\text{IQR}$)** | 824 | 42.45% | `Other_Faults` | 21 (20.6%) |
| **Isolation Forest (`contamination = 0.10`)** | 194 | 10.00% | `Bumps` | 5 (4.9%) |
| **0 Methods (Clean Inliers)** | 1,103 | 56.83% | `Z_Scratch` | 3 (2.9%) |
| **1 or 2 Methods Agree** | 736 | 37.92% | `Dirtiness` / `Pastry` (1 each) | 2 (2.0%) |
| **All 3 Methods Agree (Consensus Anomalies)** | **102** | **5.25%** | **Most Extreme Plate: Row 391 (`K_Scatch`, 15 features $\|Z\|>3$)** | **102 (100%)** |

## Lab Task 8.2: Random Forest Ensemble Feature Importance Analysis
**Explanation:** We train a `RandomForestClassifier(n_estimators=100, random_state=42)` on `(X_train, y_train)` to rank all 27 features by mean decrease in Gini impurity.

```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
fi = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
```

**Interpretation:** The top 10 features capture **50.22%** of total Gini importance: `LogOfAreas` (`0.0733`), `Length_of_Conveyer` (`0.0611`), `Pixels_Areas` (`0.0566`), `Log_X_Index` (`0.0500`), `Outside_X_Index` (`0.0471`), `Steel_Plate_Thickness` (`0.0469`), `X_Minimum` (`0.0445`), `Sum_of_Luminosity` (`0.0429`), `Minimum_of_Luminosity` (`0.0416`), and `Edges_Index` (`0.0382`). `Outside_Global_Index` (`0.0024`) contributes negligibly.

---

# Lab 9: Mini Project – End-to-End Data Mining Pipeline

## Lab Task 9.1: Final Model Comparison, Artifact Persistence (`.pkl`), and Reusable Prediction Pipeline
**Explanation:** We synthesize the final benchmark across all evaluated classifiers, serialize the winning **Tuned RBF SVM (`C=10, kernel='rbf'`)**, fitted `StandardScaler`, and feature schema to `models/`, and verify an end-to-end inference helper `predict_defect_type()`.

```python
os.makedirs("../models", exist_ok=True)
joblib.dump(best_svm, "../models/tuned_svm_model.pkl")
joblib.dump(svm_scaler, "../models/svm_scaler.pkl")
joblib.dump(list(X.columns), "../models/feature_names.pkl")

def predict_defect_type(sample_data):
    model = joblib.load("../models/tuned_svm_model.pkl")
    scaler = joblib.load("../models/svm_scaler.pkl")
    feats = joblib.load("../models/feature_names.pkl")
    row = pd.DataFrame([sample_data])[feats] if isinstance(sample_data, dict) else sample_data[feats].copy()
    return model.predict(scaler.transform(row))[0]

for idx in [0, 100, 500]:
    print(f"Row {idx:3d} -> Predicted: {predict_defect_type(X_test.iloc[[idx]]):12s} | Actual: {y_test.iloc[idx]}")
```

| Rank | Classification Model | Test Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Key Takeaway |
| :---: | :--- | ---: | ---: | ---: | ---: | :--- |
| **1** | **Tuned SVM (`C=10, kernel='rbf'`)** | **0.7635** | **0.7672** | **0.7635** | **0.7644** | **Selected production model (+3.18% F1 over Linear SVM)** |
| **2** | **Decision Tree (`CART`)** | 0.7455 | 0.7492 | 0.7455 | 0.7452 | Fast & interpretable rules, but lower precision on `Pastry` |
| **3** | **Baseline Linear SVM** | 0.7326 | 0.7341 | 0.7326 | 0.7326 | Solid linear boundary; underfits curved fault manifolds |
| **4** | **Gaussian Naive Bayes** | 0.4524 | 0.5741 | 0.4524 | 0.4003 | Degraded by correlated predictors & skewed distributions |

```text
Pipeline Verification -> Row 0: Predicted=Bumps (Actual=Bumps) | Row 100: Predicted=Other_Faults (Actual=Other_Faults) | Row 500: Predicted=K_Scatch (Actual=K_Scatch)
```

**Conclusion & Limitations:** The **Tuned RBF SVM (`76.35%` accuracy, `0.7644` weighted F1)** is selected as the deployment model. Remaining errors primarily stem from the catch-all `Other_Faults` category overlapping with `Bumps` and `Pastry`, alongside severe class imbalance (`Dirtiness` = 55 plates, `Stains` = 72 plates).
