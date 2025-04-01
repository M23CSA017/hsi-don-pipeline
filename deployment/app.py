# # # """
# # # app.py

# # # Simple FastAPI-based service that exposes a /predict endpoint for new data.
# # # """

# # # from fastapi import FastAPI, HTTPException
# # # from pydantic import BaseModel
# # # from typing import List
# # # import numpy as np
# # # import pandas as pd
# # # from src.models.predict import predict_don, predict_batch

# # # app = FastAPI()

# # # class PredictRequest(BaseModel):
# # #     features: List[float]

# # # @app.post("/predict")
# # # def predict_endpoint(req: PredictRequest):
# # #     """
# # #     Accept a list of spectral features (or PCA components)
# # #     and return the predicted DON concentration.
# # #     """
# # #     prediction = predict_don(np.array(req.features))
# # #     return {"DON_prediction": prediction}

# # # class BatchPredictRequest(BaseModel):
# # #     features: List[List[float]]

# # # @app.post("/predict_batch/")
# # # def predict_batch_api(req: BatchPredictRequest):
# # #     """
# # #     Accept a list of spectral feature lists and return predictions for each sample.
# # #     """
# # #     try:
# # #         # Convert list to DataFrame and then to numpy array
# # #         features = pd.DataFrame(req.features)
# # #         predictions = predict_batch(features.values)
# # #         return {"predictions": predictions}
# # #     except Exception as e:
# # #         raise HTTPException(status_code=500, detail=str(e))

# # """
# # app.py

# # Simple FastAPI-based service that exposes a /predict and /predict_batch endpoint for new data.
# # """

# # from fastapi import FastAPI, HTTPException
# # from pydantic import BaseModel
# # from typing import List
# # import numpy as np
# # import pandas as pd
# # from src.models.predict import predict_don, predict_batch

# # # Initialize FastAPI App
# # app = FastAPI(
# #     title="DON Concentration Prediction API",
# #     description="API to predict mycotoxin (DON) levels using spectral data.",
# #     version="1.0.0",
# # )

# # # ---------------------------
# # # 📌 Request Model for Single Prediction
# # # ---------------------------
# # class PredictRequest(BaseModel):
# #     features: List[float]


# # @app.post("/predict/")
# # def predict_endpoint(req: PredictRequest):
# #     """
# #     Accept a list of spectral features (or PCA components)
# #     and return the predicted DON concentration.
# #     """
# #     try:
# #         features_array = np.array(req.features).reshape(1, -1)
# #         prediction = predict_don(features_array)
# #         return {"DON_prediction": prediction}
# #     except Exception as e:
# #         raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")


# # # ---------------------------
# # # 📌 Request Model for Batch Prediction
# # # ---------------------------
# # class BatchPredictRequest(BaseModel):
# #     features: List[List[float]]

# # from fastapi import HTTPException

# # @app.post("/predict_batch/")
# # def predict_batch_api(payload: FeatureBatch):
# #     try:
# #         predictions = predict_batch(
# #             features_list=payload.features,
# #             model_path=MODEL_PATH,
# #             use_nn=False  # or True, based on your test config
# #         )
# #         return {"predictions": predictions}
# #     except Exception as e:
# #         print(f"🔥 Internal Server Error in /predict_batch/: {e}")
# #         raise HTTPException(status_code=500, detail=str(e))


# # # ---------------------------
# # # 📌 Health Check Endpoint
# # # ---------------------------
# # @app.get("/")
# # def health_check():
# #     """
# #     Basic health check endpoint.
# #     """
# #     return {"status": "API is running successfully"}

# """
# app.py

# Simple FastAPI-based service that exposes /predict and /predict_batch endpoints for DON prediction.
# """

# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
# import numpy as np
# import pandas as pd
# from src.models.predict import predict_don, predict_batch

# # Initialize FastAPI app
# app = FastAPI(
#     title="DON Concentration Prediction API",
#     description="API to predict mycotoxin (DON) levels using spectral data.",
#     version="1.0.0",
# )

# # ---------------------------
# # 📌 Request Model for Single Prediction
# # ---------------------------
# class PredictRequest(BaseModel):
#     features: List[float]

# @app.post("/predict/")
# def predict_endpoint(req: PredictRequest):
#     """
#     Accept a single feature vector and return predicted DON concentration.
#     """
#     try:
#         features_array = np.array(req.features).reshape(1, -1)
#         prediction = predict_don(features_array)
#         return {"DON_prediction": prediction}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

# # ---------------------------
# # 📌 Request Model for Batch Prediction
# # ---------------------------
# class BatchPredictRequest(BaseModel):
#     features: List[List[float]]

# @app.post("/predict_batch/")
# def predict_batch_api(payload: BatchPredictRequest):
#     """
#     Accept a list of feature vectors and return predictions for each sample.
#     """
#     try:
#         predictions = predict_batch(
#             features_list=payload.features,
#             model_path="./data/models/best_xgboost_model.pkl",  # or NN model
#             use_nn=False  # toggle this based on your testing config
#         )
#         return {"predictions": predictions}
#     except Exception as e:
#         print(f"🔥 Internal Server Error in /predict_batch/: {e}")
#         raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# # ---------------------------
# # ✅ Health Check
# # ---------------------------
# @app.get("/")
# def health_check():
#     return {"status": "API is running"}

"""
app.py

FastAPI service that exposes /predict and /predict_batch endpoints for DON prediction
using both XGBoost and Neural Network models.
"""

from fastapi import FastAPI, HTTPException
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
# 📌 Single Prediction Request Schema
# ---------------------------
class PredictRequest(BaseModel):
    features: List[float]

# ---------------------------
# 📌 Batch Prediction Request Schema
# ---------------------------
class BatchPredictRequest(BaseModel):
    features: List[List[float]]

# ---------------------------
# 🎯 Single Prediction Endpoint
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
# 📦 Batch Prediction Endpoint
# ---------------------------
@app.post("/predict_batch/")
def predict_batch_api(req: BatchPredictRequest):
    """
    Accept a list of feature vectors and return predictions from both models.
    """
    try:
        result = predict_batch_with_both_models(req.features)
        return result
    except Exception as e:
        print(f"🔥 Internal Server Error in /predict_batch/: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# ---------------------------
# ✅ Health Check
# ---------------------------
@app.get("/")
def health_check():
    return {"status": "API is running"}
