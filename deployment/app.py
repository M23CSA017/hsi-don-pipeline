"""
app.py

Simple FastAPI-based service that exposes a /predict endpoint for new data.
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from ..src.models.predict import predict_don

app = FastAPI()

class PredictRequest(BaseModel):
    features: List[float]

@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    """
    Accept a list of spectral features (or PCA components)
    and return the predicted DON concentration.
    """
    prediction = predict_don(np.array(req.features))
    return {"DON_prediction": prediction}
