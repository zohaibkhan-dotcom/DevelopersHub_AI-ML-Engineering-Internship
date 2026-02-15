import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import os

# Create a simple model: Predicts if a customer churns based on 2 features
# Features: [Usage Frequency, Support Tickets]
X = np.array([[10, 1], [2, 5], [15, 0], [1, 8]])
y = np.array([0, 1, 0, 1]) # 0 = Stay, 1 = Churn

model = LogisticRegression()
model.fit(X, y)

# Save the model
model_path = os.path.join(
    os.path.dirname(__file__),
    "models",
    "v1",
    "model.joblib"
)
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
