# AI/ML Engineering Internship – Task 3  
**DevelopersHub Corporation**

## 📌 Task 3: Heart Disease Prediction

---

## 🎯 Objective
The objective of this task is to develop a **machine learning classification model** that predicts whether a person is at risk of heart disease based on clinical and health-related features. This project demonstrates the complete ML workflow, including data preprocessing, visualization, model training, and evaluation.

---

## 📊 Dataset
- **Name:** Heart Disease UCI Dataset  
- **File:** `heart.csv`  
- **Source:** UCI Machine Learning Repository  

### Features
- Age  
- Sex  
- Chest Pain Type (`cp`)  
- Resting Blood Pressure (`trestbps`)  
- Serum Cholesterol (`chol`)  
- Fasting Blood Sugar (`fbs`)  
- Resting ECG (`restecg`)  
- Maximum Heart Rate Achieved (`thalach`)  
- Exercise Induced Angina (`exang`)  
- ST Depression (`oldpeak`)  
- Slope of ST Segment (`slope`)  
- Number of Major Vessels (`ca`)  
- Thalassemia (`thal`)  

### Target Variable
- **Heart Disease**
  - `0` → No heart disease  
  - `1` → Presence of heart disease  

---

## 🛠️ Tools & Technologies Used
- **Python**
- **Pandas & NumPy** – Data loading, cleaning, and preprocessing  
- **Scikit-learn** – Logistic Regression, train-test split, evaluation metrics  
- **Matplotlib & Seaborn** – Data visualization, Confusion Matrix, ROC Curve  
- **Jupyter Notebook** – Implementation and analysis  

---

## 🔍 Exploratory Data Analysis (EDA)
- Distribution analysis of key features  
- Correlation heatmap  
- Visualization of target class distribution  
- Feature relationship analysis  

---

## 🤖 Model Used
- **Logistic Regression**
  - Chosen for its simplicity and interpretability
  - Suitable for binary classification problems

---

## 📈 Model Evaluation
- **Accuracy Score**  
- **Confusion Matrix**  
- **ROC Curve & AUC Score**

---

## 🧠 Key Results & Findings
- The model achieved **high accuracy** on the test dataset.
- The **ROC-AUC curve** indicates strong predictive performance.
- The most influential features in predicting heart disease were:
  - Chest Pain Type (`cp`)
  - Number of Major Vessels (`ca`)
  - ST Depression (`oldpeak`)

---

## ▶️ How to Run the Project
1. Install Python (version 3.8 or higher recommended)
2. Install required libraries:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
