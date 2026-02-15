# Task 3: Multimodal Machine Learning – Housing Price Prediction

## 🎯 Project Objective
Build a multimodal regression system that predicts housing prices by fusing structured tabular data with visual features extracted from house images using a Convolutional Neural Network (CNN).

## 📊 Dataset Description
- **Tabular Data (`housing_data.csv`)**: Contains features such as bedrooms, bathrooms, square footage, condition, and zipcode.
- **Image Data (`images/`)**: JPEG images of houses named by their corresponding `id` in the tabular dataset.

## 🏗 Model Architecture
The system uses a late-fusion approach:
1. **CNN Branch**: A pretrained **ResNet18** (frozen) extracts a 512-dimensional feature vector from house images.
2. **Tabular Branch**: Numerical features are scaled, and categorical features are one-hot encoded.
3. **Fusion Layer**: The image feature vector and processed tabular features are concatenated.
4. **Regression Head**: A feedforward neural network (Linear -> ReLU -> Dropout -> Linear -> ReLU -> Linear) predicts the final price.

```text
[ Image ] -> [ ResNet18 ] -> [ 512-d Vector ] \
                                               > [ Concatenation ] -> [ MLP Regressor ] -> [ Price ]
[ Tabular] -> [ Preprocessor ] -> [ N-d Vector ] /
```

## 🚀 How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Generate Synthetic Data** (Optional, if dataset is not provided):
   ```bash
   python generate_data.py
   ```
3. **Run Training Pipeline**:
   ```bash
   python multimodal_pipeline.py
   ```
4. **Explore the Notebook**:
   Open `notebook.ipynb` in Jupyter to see the EDA and detailed analysis.

## 📈 Evaluation Results
- **Mean Absolute Error (MAE)**: $946,960.81
- **Root Mean Squared Error (RMSE)**: $1,003,393.23
*(Note: These values are based on the synthetic dataset generated for demonstration.)*

## 💡 Key Insights
- **Multimodal Benefits**: Integrating visual data allows the model to capture "curb appeal" and condition factors that are difficult to quantify in tabular form.
- **Modularity**: The pipeline is designed with separate modules for preprocessing, feature extraction, and training, making it easy to swap models (e.g., using EfficientNet instead of ResNet).

## 🛠 Future Improvements
- **Fine-tuning**: Unfreeze the CNN layers during the later stages of training to specialize the feature extraction for architectural styles.
- **Attention Mechanism**: Implement a cross-modal attention layer to let the model learn which tabular features correlate most strongly with visual patterns.
- **Hyperparameter Optimization**: Use Optuna or GridSearch to find the optimal architecture for the regression head.

## 📦 Production Deployment Notes
- **Model Versioning**: Use tools like MLflow or DVC to track model versions and dataset iterations.
- **Feature Store**: In production, tabular features should be served from a feature store for low-latency inference.
- **API Serving**: The model can be wrapped in a FastAPI service, accepting a JSON payload for tabular data and a multipart/form-data for the image.
