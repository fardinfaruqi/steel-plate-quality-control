# Steel Plate Quality Control — Real Application Roadmap

> Reference document for the implementation phase after completing the data mining and analysis work.

## 1. Project Goal

Build a real machine-learning application that accepts measurements/features of a steel plate and predicts its defect type using the classification model developed during the data mining phase.

The application should follow this flow:

```text
User Input
    ↓
Input Validation
    ↓
Same Preprocessing Used During Training
    ↓
Trained Classification Model
    ↓
Prediction
    ↓
Predicted Defect Type
    ↓
Confidence / Probability (if supported)
```

The existing notebook remains the main place for experimentation, analysis, model comparison, and documentation. The application should use the final selected model from that work.

---

## 2. Current Project Structure

Current structure:

```text
steel-plate-quality-control/
│
├── .venv/
├── data/
│   ├── Faults.NNA
│   └── Faults27x7_var
│
├── notebooks/
│   └── steel_plate_quality_control.ipynb
│
├── outputs/
│   ├── figures/
│   └── models/
│
├── reports/
├── requirements.txt
└── README.md
```

The real application can be added without destroying this structure.

Recommended future structure:

```text
steel-plate-quality-control/
│
├── .venv/
│
├── data/
│   ├── Faults.NNA
│   └── Faults27x7_var
│
├── notebooks/
│   └── steel_plate_quality_control.ipynb
│
├── outputs/
│   ├── figures/
│   └── models/
│       ├── best_model.joblib
│       ├── preprocessing.joblib
│       └── label_encoder.joblib
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app/
│   └── app.py
│
├── reports/
│
├── requirements.txt
├── README.md
└── REAL_APPLICATION_ROADMAP.md
```

The exact structure can be changed later depending on whether the application uses Streamlit, Flask, FastAPI, or another framework.

---

# 3. What Must Be Finished Before Deployment

Do not start deployment immediately after EDA.

First complete the data mining workflow:

1. Dataset inspection
2. Missing-value analysis
3. Duplicate analysis
4. Categorical encoding
5. Feature scaling where appropriate
6. Descriptive statistics
7. EDA
8. Classification
   - Decision Tree
   - Naive Bayes
   - SVM
   - Model comparison
9. Clustering
10. Optional Random Forest
11. Overall findings
12. Conclusion

Then decide which classification model should become the production model.

The production model should be selected based on more than just one accuracy number.

Consider:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Cross-validation performance
- Training/inference speed
- Model complexity
- Interpretability
- Whether the model provides useful prediction probabilities

For a multi-class defect-classification problem, pay particular attention to **per-class recall and F1-score**, because a high overall accuracy can hide poor performance on individual defect classes.

---

# 4. Critical Rule: Reuse the Exact Training Preprocessing

This is one of the most important parts of the entire application.

The data given to the production model must be processed in the same way as the training data.

For example, if training used:

```text
Raw features
    ↓
Categorical encoding
    ↓
Feature scaling
    ↓
Model
```

then prediction must use:

```text
New user input
    ↓
Same categorical encoding
    ↓
Same scaler fitted during training
    ↓
Model
```

## Never do this

Do not fit a new scaler on the user's input.

Wrong:

```python
scaler = StandardScaler()
new_data = scaler.fit_transform(new_data)
```

The scaler used in production must have been fitted on the training data.

Correct:

```python
scaler = joblib.load("outputs/models/preprocessing.joblib")
new_data = scaler.transform(new_data)
```

The same principle applies to:

- Label encoding
- One-hot encoding
- Feature selection
- Imputation
- Transformation
- Column ordering

---

# 5. Prefer a Single Preprocessing Pipeline

Instead of manually saving many preprocessing objects, consider using a scikit-learn `Pipeline` and/or `ColumnTransformer`.

Conceptually:

```text
Raw Input
   ↓
Preprocessing Pipeline
   ├── Missing-value handling
   ├── Encoding
   ├── Scaling
   └── Feature selection/transformation
   ↓
Classifier
   ↓
Prediction
```

A strong production approach is to save the complete pipeline:

```python
joblib.dump(model_pipeline, "outputs/models/best_model.joblib")
```

Then prediction becomes:

```python
model = joblib.load("outputs/models/best_model.joblib")
prediction = model.predict(new_data)
```

This reduces the risk of accidentally applying different preprocessing during deployment.

---

# 6. Saving the Final Model

After selecting the final classifier, save it using a serialization library such as `joblib`.

Example:

```python
import joblib

joblib.dump(
    model_pipeline,
    "outputs/models/best_model.joblib"
)
```

Also save any separate objects that are genuinely required, such as a label encoder:

```python
joblib.dump(
    label_encoder,
    "outputs/models/label_encoder.joblib"
)
```

The exact files should depend on how the final training pipeline is implemented.

---

# 7. Create a Prediction Function

Create a reusable prediction module rather than putting all prediction logic directly inside the UI.

Example concept:

```python
def predict_defect(input_data):
    model = load_model()

    prediction = model.predict(input_data)

    return prediction
```

If the selected classifier supports probabilities:

```python
probabilities = model.predict_proba(input_data)
```

The prediction layer should be independent of the user interface.

That means:

```text
UI
 ↓
Prediction Function
 ↓
Model
```

rather than:

```text
UI
 ↓
50 lines of preprocessing
 ↓
50 lines of model code
 ↓
Prediction
```

This makes the application easier to test and maintain.

---

# 8. Define the Input Features Carefully

The application needs to know exactly which features are required to make a prediction.

Create a definitive feature list based on the final training dataset.

Example:

```python
FEATURE_COLUMNS = [
    "feature_1",
    "feature_2",
    "feature_3",
    # ...
]
```

The application must:

- Accept all required features
- Use the correct feature names
- Maintain the correct feature order
- Validate numeric inputs
- Reject missing required values
- Prevent invalid values where possible

Do not rely on the order in which a user happens to enter the values.

---

# 9. Design the User Interface

A simple first version is enough.

For example:

```text
┌──────────────────────────────────────────────┐
│       Steel Plate Quality Control             │
│                                              │
│  Enter plate measurements                    │
│                                              │
│  Feature 1: [____________]                   │
│  Feature 2: [____________]                   │
│  Feature 3: [____________]                   │
│  Feature 4: [____________]                   │
│  ...                                         │
│                                              │
│              [ Predict Defect ]              │
│                                              │
├──────────────────────────────────────────────┤
│  Prediction                                  │
│                                              │
│  Defect Type: K-Scratch                      │
│  Confidence: 91.4%                           │
└──────────────────────────────────────────────┘
```

For a course project, **Streamlit** is a practical first choice because it allows a Python-based ML model to be turned into an interactive web application quickly.

A more production-oriented architecture could later use:

```text
Frontend
   ↓
REST API
   ↓
ML Model
```

using a framework such as FastAPI.

---

# 10. Suggested Application Architecture

## Simple Version

Recommended for the first implementation:

```text
Streamlit UI
     ↓
Prediction Module
     ↓
Saved ML Pipeline
     ↓
Prediction
```

Project structure:

```text
app/
└── app.py

src/
├── preprocessing.py
└── predict.py

outputs/models/
└── best_model.joblib
```

## More Advanced Version

If the project is later expanded:

```text
React Frontend
       ↓
FastAPI Backend
       ↓
Prediction Service
       ↓
Saved ML Model
```

This is useful if you want to demonstrate full-stack development in addition to machine learning.

---

# 11. Prediction Output

The minimum output should be:

```text
Predicted Defect: <class>
```

A better version:

```text
Predicted Defect: <class>
Confidence: <percentage>
```

An even better version, when `predict_proba()` is available:

```text
Prediction:
    K-Scratch      91.4%
    Z-Scratch       4.2%
    Bumps           2.1%
    Other           2.3%
```

This makes the application more informative.

Important: confidence/probability should only be shown if the model's probability output is appropriate. Do not describe a model probability as a guaranteed real-world probability.

---

# 12. Add Input Validation

Before sending input to the model, validate:

### Required fields

Every required feature must be present.

### Numeric values

Numeric features should reject:

```text
abc
hello
---
```

### Range validation

Where domain knowledge or dataset constraints justify it, reject clearly invalid values.

For example:

```text
Area < 0
Perimeter < 0
```

should not normally be accepted.

However, do not invent arbitrary limits simply because they appear in the training dataset. Dataset minimum/maximum values are not automatically valid physical limits.

---

# 13. Test the Application

Before considering the application complete, test at least these cases:

### Test 1 — Valid input

Give a valid sample and verify that a prediction is returned.

### Test 2 — Missing input

Leave a required field empty.

Expected:

```text
Please provide all required values.
```

### Test 3 — Invalid input

Enter text where a numeric value is required.

Expected:

```text
Invalid value. Please enter a number.
```

### Test 4 — Multiple samples

If batch prediction is implemented, upload a CSV and verify predictions for every row.

### Test 5 — Known dataset samples

Take samples from the test set that were not used to train the model and run them through the application.

Compare:

```text
Actual Label
vs.
Application Prediction
```

This is an important validation step.

---

# 14. Avoid Data Leakage

The final application must never use the test set to train the model.

Correct workflow:

```text
Complete Dataset
       ↓
Train/Test Split
       ↓
Training Data ──→ Fit preprocessing + model
       │
       ↓
Test Data ──────→ Evaluate
```

The test data should remain unseen until evaluation.

When deploying the model, save the preprocessing/model pipeline that was fitted using the training data.

---

# 15. Optional Batch Prediction

After the single-input version works, add CSV upload.

Example:

```text
[ Upload CSV ]

        ↓

Validate columns

        ↓

Preprocess rows

        ↓

Predict all rows

        ↓

Download results
```

Output:

```text
input.csv
     ↓
predictions.csv
```

Example:

```text
Feature_1,Feature_2,...,Predicted_Defect
42.1,15.3,...,K-Scratch
35.8,21.4,...,Bumps
...
```

This would make the application much closer to a practical quality-control tool.

---

# 16. Optional Prediction History

A later version could store prediction records:

```text
Timestamp
Input Features
Predicted Defect
Confidence
```

Possible architecture:

```text
Application
     ↓
Prediction
     ↓
Database
```

For the first version, a database is not necessary.

---

# 17. Optional Explainability

If the final model supports useful feature interpretation, add an explanation section.

For example:

```text
Prediction: K-Scratch

Important contributing features:
1. Feature A
2. Feature B
3. Feature C
```

For tree-based models, feature importance can be useful.

For more advanced explainability, SHAP can be considered later.

Do not add explainability just for decoration; the explanation method should be technically appropriate for the chosen model.

---

# 18. Model Versioning

When experimenting with different models, avoid overwriting everything blindly.

For example:

```text
outputs/models/
├── decision_tree.joblib
├── naive_bayes.joblib
├── svm.joblib
├── random_forest.joblib
└── best_model.joblib
```

Then record the final model information in a small metadata file if useful:

```text
Model:
Random Forest

Validation Accuracy:
XX.XX%

Test Accuracy:
XX.XX%

F1 Score:
XX.XX%

Training Date:
YYYY-MM-DD
```

The exact metrics should come from the final experiment rather than being hard-coded in advance.

---

# 19. Requirements

Once deployment begins, update `requirements.txt`.

Possible packages:

```text
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
streamlit
```

Only keep packages that are actually used by the final application.

If a separate API is created:

```text
fastapi
uvicorn
```

may also be required.

---

# 20. Recommended Implementation Order

Do the real application in this order:

```text
Step 1
Finish data mining notebook
        ↓
Step 2
Select final classification model
        ↓
Step 3
Build final preprocessing pipeline
        ↓
Step 4
Train model on training data
        ↓
Step 5
Evaluate on untouched test data
        ↓
Step 6
Save model/pipeline
        ↓
Step 7
Create prediction.py
        ↓
Step 8
Test prediction from Python
        ↓
Step 9
Build simple Streamlit UI
        ↓
Step 10
Connect UI to prediction function
        ↓
Step 11
Add validation and error handling
        ↓
Step 12
Test with unseen samples
        ↓
Step 13
Add batch CSV prediction (optional)
        ↓
Step 14
Improve UI/documentation
        ↓
Step 15
Deploy
```

---

# 21. Deployment

Once the local application works:

```text
Local ML Application
        ↓
Git Repository
        ↓
Cloud Deployment
        ↓
Public URL
```

For a simple Streamlit application, a suitable hosting platform can be selected later based on current availability, cost, and project requirements.

Do not deploy before verifying that:

- The model loads correctly
- The preprocessing is consistent
- Predictions work
- Invalid input is handled
- The requirements file is complete
- The application works without the notebook

The production application should not depend on running the notebook.

---

# 22. Important Principle

The notebook is for:

```text
Experimentation
Analysis
Visualization
Model comparison
Research findings
```

The application is for:

```text
Prediction
User interaction
Validation
Model inference
Practical demonstration
```

Do not copy the entire notebook into the application.

The notebook should answer:

> "How did we develop and evaluate this model?"

The application should answer:

> "Can I give this system new steel-plate measurements and get a prediction?"

---

# 23. Final Target

The completed project should ideally demonstrate both:

### Academic / Data Mining Component

```text
Dataset
→ Preprocessing
→ Statistics
→ EDA
→ Classification
→ Clustering
→ Model Comparison
→ Findings
→ Conclusion
```

### Practical ML Application

```text
New Steel Plate Data
→ Validation
→ Preprocessing
→ Trained Model
→ Defect Prediction
→ Confidence / Probability
→ Result
```

This gives the project a clear progression from **data analysis → machine learning → usable application**.

---

# 24. Final Checklist

Before calling the project complete:

- [ ] Data mining notebook completed
- [ ] All required EDA completed
- [ ] Classification models evaluated
- [ ] Best model selected using appropriate metrics
- [ ] Test set kept separate from training
- [ ] Final preprocessing pipeline created
- [ ] Model saved
- [ ] Prediction function created
- [ ] New input successfully produces predictions
- [ ] Input validation implemented
- [ ] User interface implemented
- [ ] Application tested with unseen data
- [ ] Optional batch prediction implemented
- [ ] `requirements.txt` updated
- [ ] README updated
- [ ] Application deployed (if required)

---

## 25. First Task When Returning to This Document

When the data mining part is finished, start here:

1. Review the model comparison results.
2. Identify the best classification model.
3. Check exactly which preprocessing steps were applied.
4. Refactor those preprocessing steps into a reusable pipeline.
5. Train the final pipeline correctly.
6. Save it under `outputs/models/`.
7. Create `src/predict.py`.
8. Test predictions without using the notebook.
9. Build the UI.
10. Connect the UI to the prediction module.

**Do not start by building the UI. Build and verify the model inference pipeline first.**
