# Steel Plate Quality Control

Course mini-project: *A Data Mining Approach to Manufacturing Quality Control: Classification and Analysis of Steel Plate Defects*.

This repository currently contains only the isolated Python environment and project structure. No data-mining analysis has been added yet.

## Folders

- `data/` — source dataset files
- `notebooks/` — Jupyter notebooks
- `outputs/figures/` — generated plots
- `outputs/models/` — saved model artifacts
- `reports/` — written project reports

## Environment

The project uses `.venv` and the Jupyter kernel **Steel Plate Quality Control**.


## Actual Project Report Structure:
```txt
1. Introduction
2. Dataset Description
3. Import Libraries
4. Load Dataset
5. Dataset Inspection
6. Data Preprocessing
   ├── Missing Values
   ├── Duplicates
   ├── Categorical Encoding
   └── Feature Scaling
7. Descriptive Statistics
8. Exploratory Data Analysis
   ├── Histograms
   ├── Bar Charts
   ├── Scatter Plots
   ├── Box Plots
   ├── Correlation Heatmap
   └── Outlier Analysis
9. Classification
   ├── Decision Tree
   ├── Naive Bayes
   ├── SVM
   └── Model Comparison
10. Clustering
    ├── Elbow Method
    ├── K-Means
    └── Cluster Interpretation
11. Optional Random Forest
    └── Feature Importance
12. Overall Findings
13. Conclusion
```


## Current project structure:
```txt
# 1. Import Libraries

# 2. Load Dataset

# 3. Basic Dataset Inspection

# 4. Data Understanding
├──## 4.1 Missing Values
├──## 4.2 Duplicate Rows
├──## 4.3 Data Types
├──## 4.4 Descriptive Statistics
├──## 4.5 Defect Class Distribution
└──## 4.6 Target Variable

# 5. Data Preprocessing
├──## 5.1 Separate Features and Target
├──## 5.2 Check Feature Scales
└──## 5.3 Feature Scaling

# 6. Exploratory Data Analysis (EDA)
├──## 6.1 Defect Distribution
├──## 6.2 Histograms
├──## 6.3 Bar Chart
├──## 6.4 Scatter Plot
├──## 6.5 Box Plot
├──## 6.6 Correlation Matrix and Heatmap
└──## 6.7 Outlier and Relationship Analysis

# 7. Classification   ← NOW
├──## 7.1 Train-Test Split
├──## 7.2 Decision Tree
├──## 7.3 Naive Bayes
├──## 7.4 Support Vector Machine (SVM)
├──## 7.5 Model Evaluation
└──## 7.6 Classification Model Comparison

# 8. Clustering
├──## 8.1 Elbow Method
├──## 8.2 K-Means Clustering
└──## 8.3 Cluster Analysis

# 9. Outlier Detection and Feature Importance
├──## 9.1 Outlier Detection
├──## 9.2 Random Forest
└──## 9.3 Feature Importance

# 10. Overall Results and Findings

# 11. Conclusion
```