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
|   ├──### 7.2.1 Train Model 
|   ├──### 7.2.2 Make Predictions
|   ├──### 7.2.3 Accuracy
|   ├──### 7.2.4 Classification Report
|   ├──### 7.2.5 Confusion Matrix
|   └──### 7.2.6 Tree Visualization
├──## 7.3 Naive Bayes
|   ├──### 7.3.1 Train Model
|   ├──### 7.3.2 Make Predictions
|   ├──### 7.3.3 Accuracy
|   ├──### 7.3.4 Classification Report
|   └──### 7.3.5 Confusion Matrix
├──## 7.4 Support Vector Machine (SVM)
|   ├──### 7.4.1 Scale Training/Test Data
|   ├──### 7.4.2 Train Linear SVM
|   ├──### 7.4.3 Make Predictions
|   ├──### 7.4.4 Accuracy
|   ├──### 7.4.5 Classification Report
|   └──### 7.4.6 Confusion Matrix
├──## 7.5 Model Evaluation     ← NOW
|   ├──### 7.5.1 Evaluation Metrics
|   ├──### 7.5.2 Comparison Table
|   ├──### 7.5.3 Rounded Results
|   ├──### 7.5.4 Visual Comparison
|   └──### 7.5.5 Best Baseline Model
├──## 7.6 Classification Model Comparison
|   ├──### 7.6.1 Display the Final Baseline Comparison
|   ├──### 7.6.2 Rank the Models
|   ├──### 7.6.3 Identify the Best Baseline Model
|   ├──### 7.6.4 Compare the Models Visually
|   ├──### 7.6.5 Check the Individual Defect Performance
|   └──### 7.6.6 Find the Best Model for Each Defect
├──## 7.7 SVM Hyperparameter Tuning
|   ├──### 7.7.1 Define Parameters
|   ├──### 7.7.2 Grid Search
|   ├──### 7.7.3 Best Parameters
|   ├──### 7.7.4 Grid Search Results
|   ├──### 7.7.5 Best SVM
|   └──### 7.7.6 Tuned SVM Predictions
└──## 7.8 Tuned SVM Evaluation
|   ├──### 7.8.1 Tuned SVM Accuracy
|   ├──### 7.8.2 Classification Report
|   ├──### 7.8.3 Confusion Matrix
|   └──### 7.8.4 Compare Baseline vs Tuned SVM
└──## 7.9 Final Classification Results

# 8. Clustering
├──## 8.1 Prepare Data for Clustering
├──## 8.2 K-Means Clustering
|   ├──### 8.2.1 Import K-Means
|   ├──### 8.2.2 Create K-Means Model
|   ├──### 8.2.3 Get Cluster Labels
|   ├──### 8.2.4 Count Cluster Observations
|   └──### 8.2.5 Calculate Inertia
├──## 8.3 Elbow Method
|   ├──### 8.3.1 Calculate Inertia
|   ├──### 8.3.2 Display Results
|   ├──### 8.3.3 Plot Elbow Curve
|   └──### 8.3.4 Select K
├──## 8.4 Choose Number of Clusters
├──## 8.5 Train Final K-Means Model
|   ├──### 8.5.1 Final Cluster Labels
|   ├──### 8.5.2 Cluster Counts
|   └──### 8.5.3 Add Labels to Dataset
├──## 8.6 Visualize Clusters
|   ├──### 8.6.1 Import PCA
|   ├──### 8.6.2 PCA Transformation
|   ├──### 8.6.3 Explained Variance
|   ├──### 8.6.4 Cluster Visualization
|   └──### 8.6.5 Cluster Centers
└──## 8.7 Interpret Clusters
|   ├──### 8.7.1 Cluster–Defect Table
|   ├──### 8.7.2 Defect Counts
|   ├──### 8.7.3 Defect Percentages
|   ├──### 8.7.4 Heatmap
|   ├──### 8.7.5 Dominant Defects
|   └──### 8.7.6 Small Cluster Investigation
└──## 8.8 Clustering Summary             ← NOW

# 9. Outlier Detection and Feature Importance
├──## 9.1 Outlier Detection
├──## 9.2 Random Forest
└──## 9.3 Feature Importance

# 10. Overall Results and Findings

# 11. Conclusion
```