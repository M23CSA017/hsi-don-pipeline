"""
predict.py

Provides a function to generate predictions from a trained model.
Useful for API integration or batch inference.
"""

import numpy as np
import joblib
import logging

final_model = None

def predict_don(features,
                model_path: str = '../data/models/best_xgboost_model.pkl') -> float:
    """
    Predict DON concentration from spectral features.

    :param features: 1D array-like of spectral features / PCA components
    :param model_path: path to the saved XGB model
    :return: predicted DON level (float)
    """
    logger = logging.getLogger(__name__)
    global final_model
    try:
        if final_model is None:
            final_model = joblib.load(model_path)
        features = np.array(features).reshape(1, -1)
        prediction = float(final_model.predict(features)[0])
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise e
