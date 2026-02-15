# Task 2: End-to-End ML Pipeline for Customer Churn Prediction

## Project Objective
The goal of this project is to build a production-ready machine learning pipeline to predict customer churn for a telecommunications company. The pipeline automates the entire process from raw data preprocessing to model prediction and export.

## Dataset Description
The **Telco Customer Churn** dataset contains information about a fictional telco company that provided home phone and Internet services to 7,043 customers.
- **Features:** Demographic information (gender, seniority, etc.), services (phone, internet, streaming, etc.), and account information (tenure, contract, payment method).
- **Target:** `Churn` (Whether the customer left within the last month).

## Pipeline & Approach
1. **Data Preprocessing:**
   - **Numerical Features:** Handled missing values in `TotalCharges` and applied `StandardScaler`.
   - **Categorical Features:** Encoded using `OneHotEncoder` within the `ColumnTransformer`.
2. **Modeling:**
   - Evaluated **Logistic Regression** and **Random Forest Classifier**.
   - Used `GridSearchCV` for hyperparameter tuning to optimize the **F1-macro score**.
3. **Encapsulation:**
   - The entire workflow is wrapped in a `scikit-learn Pipeline`, ensuring that preprocessing is consistently applied to both training and inference data.

## Model Training & Evaluation
The models were evaluated using Accuracy and F1-score (macro).
- **Logistic Regression:** Good baseline, highly interpretable.
- **Random Forest:** Handles non-linear relationships and feature interactions effectively.

## How to run the pipeline
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Script:**
   ```bash
   python churn_pipeline.py
   ```
3. **Explore the Notebook:**
   Open `notebook.ipynb` in Jupyter for detailed EDA and step-by-step walkthrough.

## Key Results / Observations
- The `TotalCharges` column required special handling due to empty strings.
- Random Forest generally performs well, but Logistic Regression provides a strong, simple baseline.
- The final exported `churn_pipeline.joblib` can be reloaded to make predictions on new raw data without manual preprocessing.

## Deployment / Reuse
To reuse the pipeline in a production environment:
```python
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load('churn_pipeline.joblib')

# Predict on new data (must be a DataFrame with original column names)
# new_predictions = pipeline.predict(new_df)
```
