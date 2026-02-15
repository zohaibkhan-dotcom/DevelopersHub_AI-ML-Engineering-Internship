from pydantic import BaseModel
from typing import List

class PredictionRequest(BaseModel):
    # Features: [Usage Frequency, Support Tickets]
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: List[float]
    status: str = "success"

class HealthResponse(BaseModel):
    status: str
    model_version: str
