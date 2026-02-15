import joblib
import os
from app.config import settings
from app.logging_config import logger

class ModelLoader:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            if os.path.exists(settings.MODEL_PATH):
                self.model = joblib.load(settings.MODEL_PATH)
                logger.info(f"Model loaded from {settings.MODEL_PATH}")
            else:
                logger.error(f"Model file not found at {settings.MODEL_PATH}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e

    def predict(self, features):
        if self.model is None:
            raise Exception("Model not loaded")
        
        # Expecting features as a list [f1, f2]
        import numpy as np
        data = np.array(features).reshape(1, -1)
        prediction = self.model.predict(data)[0]
        probability = self.model.predict_proba(data)[0].tolist()
        
        return int(prediction), probability

model_manager = ModelLoader()
