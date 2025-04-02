"""
app.py

FastAPI service that exposes /predict and /predict_batch endpoints for DON prediction
using both XGBoost and Neural Network models.
"""

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import numpy as np
from src.models.predict import (
    predict_with_both_models,
    predict_batch_with_both_models,
)

# Initialize FastAPI app
app = FastAPI(
    title="DON Concentration Prediction API",
    description="API to predict mycotoxin (DON) levels using spectral reflectance or PCA-transformed data.",
    version="1.0.0",
)

# ---------------------------
# Single Prediction Request Schema
# ---------------------------
class PredictRequest(BaseModel):
    features: List[float]

# ---------------------------
#  Batch Prediction Request Schema
# ---------------------------
class BatchPredictRequest(BaseModel):
    features: List[List[float]]

# ---------------------------
# Single Prediction Endpoint
# ---------------------------
@app.post("/predict/")
def predict_endpoint(req: PredictRequest):
    """
    Accept a single feature vector and return predictions from both models.
    """
    try:
        result = predict_with_both_models(req.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

# ---------------------------
# Batch Prediction Endpoint
# ---------------------------
@app.post("/predict_batch/")
async def predict_batch_api(req: BatchPredictRequest):
    """
    Accept a list of feature vectors and return predictions from both models.
    The response is wrapped in a "predictions" key as a list of dictionaries,
    each containing predictions from XGBoost and the Neural Network.
    """
    try:
        result = predict_batch_with_both_models(req.features)
        # Combine individual model predictions into a unified list
        xgb_preds = result.get("xgboost_predictions")
        nn_preds = result.get("nn_predictions")
        if xgb_preds is None or nn_preds is None or len(xgb_preds) != len(nn_preds):
            raise ValueError("Prediction outputs are missing or have mismatched lengths.")
        
        combined_predictions = []
        for xgb_pred, nn_pred in zip(xgb_preds, nn_preds):
            combined_predictions.append({"xgboost": xgb_pred, "nn": nn_pred})
        
        return {"predictions": combined_predictions}
    except Exception as e:
        print(f"Internal Server Error in /predict_batch/: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# ---------------------------
# Health Check
# ---------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}
