# AI/ML Engineering Internship - Task 6

This repository contains the sixth task for the AI/ML Engineering Internship at DevelopersHub Corporation.

## Task 6: House Price Prediction

### Objective
The goal of this task is to predict house prices using property features such as size, bedrooms, and location.

### Dataset
- **Name**: House Price Prediction Dataset
- **Features**: Square Footage, Bedrooms, Bathrooms, Location.
- **Target**: Price.

### Models/Tools Applied
- **Pandas & NumPy**: For data handling and preprocessing.
- **Scikit-Learn**: For Label Encoding, Data Splitting, and Linear Regression modeling.
- **Matplotlib & Seaborn**: For visualizing actual vs predicted prices.

### Key Results and Findings
- **Mean Absolute Error (MAE)**: Calculated to measure the average magnitude of errors in predictions.
- **Root Mean Squared Error (RMSE)**: Calculated to measure the standard deviation of the prediction errors.
- The model shows a strong correlation between square footage and price, with location also playing a significant role.

### How to Run
1. Ensure you have Python installed.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn jupyter`
3. Ensure `house_prices.csv` is in the same directory as the notebook.
4. Open the notebook: `jupyter notebook Task6_House_Price_Prediction.ipynb`
