import os

class Config:
    PROJECT_NAME = "Production-ML-Deployment"
    VERSION = "v1"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", VERSION, "model.joblib")
    LOG_FILE = "app.log"
    DEBUG = True

settings = Config()
