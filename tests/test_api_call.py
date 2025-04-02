import requests
import numpy as np


import pytest
import os

# Skip this test file in CI/CD where server is not running
if os.getenv("CI", "false") == "true":
    pytest.skip("Skipping external API call tests during CI", allow_module_level=True)


API_BASE = "http://127.0.0.1:8000"


def test_single_prediction():
    features = np.random.rand(70).tolist() 
    response = requests.post(f"{API_BASE}/predict/", json={"features": features})

    if response.status_code == 200:
        result = response.json()
        print("Single Prediction Results")
        print("XGBoost Prediction:", result.get("xgboost_prediction"))
        print("NN Prediction:", result.get("nn_prediction"))
    else:
        print("Error in single prediction:", response.text)

def test_batch_prediction():
    batch = np.random.rand(3, 70).tolist()
    response = requests.post(f"{API_BASE}/predict_batch/", json={"features": batch})

    if response.status_code == 200:
        result = response.json()
        print("Batch Prediction Results")
        print("XGBoost Predictions:", result.get("xgboost_predictions"))
        print("NN Predictions:", result.get("nn_predictions"))
    else:
        print("Error in batch prediction:", response.text)

if __name__ == "__main__":
    test_single_prediction()
    test_batch_prediction()
