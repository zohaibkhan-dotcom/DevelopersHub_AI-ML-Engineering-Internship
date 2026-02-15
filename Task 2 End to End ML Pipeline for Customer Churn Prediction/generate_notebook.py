import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# 1. Problem Statement & Objective
nb['cells'].append(nbf.v4.new_markdown_cell("""# End-to-End ML Pipeline for Customer Churn Prediction

## 1. Problem Statement & Objective
Customer churn occurs when customers stop doing business with a company. For telecommunications companies, retaining existing customers is often more cost-effective than acquiring new ones.

**Objective:**
Build a production-ready machine learning pipeline using `scikit-learn` to predict customer churn based on the Telco Customer Churn dataset. The pipeline will handle:
- Data Preprocessing (Missing values, Encoding, Scaling)
- Model Training (Logistic Regression, Random Forest)
- Hyperparameter Tuning (GridSearchCV)
- Evaluation & Selection
- Model Export for production use."""))

# 2. Dataset Loading
nb['cells'].append(nbf.v4.new_markdown_cell("""## 2. Dataset Loading
We load the Telco Customer Churn dataset. This dataset contains information about:
- **Demographics:** Gender, seniority, partner, dependents.
- **Services:** Phone, multiple lines, internet, online security, etc.
- **Account Info:** Tenure, contract, payment method, charges.
- **Target:** Churn (Yes/No)."""))

nb['cells'].append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Load dataset
df = pd.read_csv('Telco-Customer-Churn.csv')
print(f"Dataset Shape: {df.shape}")
df.head()"""))

# 3. EDA
nb['cells'].append(nbf.v4.new_markdown_cell("""## 3. Exploratory Data Analysis (EDA)
We examine the data distribution and check for missing values or data type issues."""))

nb['cells'].append(nbf.v4.new_code_cell("""# Check data types and missing values
print(df.info())

# 'TotalCharges' should be numeric, but has empty strings
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop CustomerID as it's not a feature
df.drop('customerID', axis=1, inplace=True)

# Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='viridis')
plt.title('Distribution of Customer Churn')
plt.show()

# Statistics for numerical features
print(df.describe())"""))

# 4. Data Preprocessing using Pipeline
nb['cells'].append(nbf.v4.new_markdown_cell("""## 4. Data Preprocessing using Pipeline
We define numerical and categorical features and create a `ColumnTransformer` to handle them."""))

nb['cells'].append(nbf.v4.new_code_cell("""# Split features and target
X = df.drop('Churn', axis=1)
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# Identify feature types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# Define Preprocessing
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])"""))

# 5. Model Training & 6. Hyperparameter Tuning
nb['cells'].append(nbf.v4.new_markdown_cell("""## 5. Model Training & 6. Hyperparameter Tuning
We create a main Pipeline and use `GridSearchCV` to find the best parameters for Logistic Regression and Random Forest."""))

nb['cells'].append(nbf.v4.new_code_cell("""# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 1. Logistic Regression Pipeline
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 2. Random Forest Pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter Grids
lr_param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__solver': ['liblinear', 'lbfgs']
}

rf_param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Grid Search
print("Tuning Logistic Regression...")
lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
lr_grid.fit(X_train, y_train)

print("Tuning Random Forest...")
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print(f"Best LR Params: {lr_grid.best_params_}")
print(f"Best RF Params: {rf_grid.best_params_}")"""))

# 7. Model Evaluation
nb['cells'].append(nbf.v4.new_markdown_cell("""## 7. Model Evaluation
We compare the performance of both models on the test set."""))

nb['cells'].append(nbf.v4.new_code_cell("""def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"--- {name} Evaluation ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score (Macro): {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    return acc, f1

lr_acc, lr_f1 = evaluate_model(lr_grid.best_estimator_, X_test, y_test, "Logistic Regression")
rf_acc, rf_f1 = evaluate_model(rf_grid.best_estimator_, X_test, y_test, "Random Forest")

# Compare
comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [lr_acc, rf_acc],
    'F1 Macro': [lr_f1, rf_f1]
})
print(comparison)"""))

# 8. Export Best Pipeline
nb['cells'].append(nbf.v4.new_markdown_cell("""## 8. Export Best Pipeline
We save the best performing model (based on F1 Macro) for future use."""))

nb['cells'].append(nbf.v4.new_code_cell("""best_model = rf_grid.best_estimator_ if rf_f1 > lr_f1 else lr_grid.best_estimator_
model_path = 'churn_pipeline.joblib'

joblib.dump(best_model, model_path)
print(f"Best model exported to {model_path}")

# Instructions to reload
# loaded_model = joblib.load('churn_pipeline.joblib')
# predictions = loaded_model.predict(new_data)"""))

# 9. Final Summary / Insights
nb['cells'].append(nbf.v4.new_markdown_cell("""## 9. Final Summary / Insights
- The **Logistic Regression** and **Random Forest** models showed competitive performance.
- Feature engineering (handling 'TotalCharges') and proper scaling were crucial.
- The pipeline is fully encapsulated, meaning new raw data can be passed directly to `predict()`.
- Future work could include more complex feature engineering or testing Gradient Boosting models."""))

# Save the notebook
with open('Task-2-ML-Pipeline-Churn/notebook.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully.")
