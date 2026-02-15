"""
End-to-End ML Pipeline for Customer Churn Prediction
This script handles data loading, preprocessing, training, tuning, evaluation, and export.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_data(filepath):
    """Load and perform basic cleaning on the dataset."""
    df = pd.read_csv(filepath)
    # Convert TotalCharges to numeric, handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    # Drop customerID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    return df

def build_pipeline(numeric_features, categorical_features):
    """Define the preprocessing and model pipeline structure."""
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
        ])
    
    return preprocessor

def main():
    # 1. Load Data
    print("Loading data...")
    df = load_data('Telco-Customer-Churn.csv')
    
    # 2. Split Features and Target
    X = df.drop('Churn', axis=1)
    y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Build Preprocessor
    preprocessor = build_pipeline(numeric_features, categorical_features)
    
    # 4. Define Models & Param Grids
    models = {
        'LogisticRegression': {
            'model': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))]),
            'params': {
                'classifier__C': [0.1, 1, 10],
                'classifier__solver': ['liblinear', 'lbfgs']
            }
        },
        'RandomForest': {
            'model': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42))]),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [None, 10, 20]
            }
        }
    }
    
    best_models = {}
    
    # 5. Training and Tuning
    for name, config in models.items():
        print(f"Tuning {name}...")
        grid = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1_macro', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_models[name] = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
    
    # 6. Evaluation
    results = {}
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results[name] = {'f1': f1, 'model': model}
        print(f"\n--- {name} Results ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
    # 7. Export Best Model
    best_model_name = max(results, key=lambda x: results[x]['f1'])
    best_model = results[best_model_name]['model']
    
    joblib.dump(best_model, 'churn_pipeline.joblib')
    print(f"\nBest model ({best_model_name}) exported to 'churn_pipeline.joblib'")

if __name__ == "__main__":
    main()
