
# tests/test_api.py

from fastapi.testclient import TestClient
from deployment.app import app 
import numpy as np

client = TestClient(app)

def test_batch_prediction():
    # Create two sample feature vectors, each with 70 features.
    sample_features = {
        "features": [
            np.random.rand(70).tolist(),
            np.random.rand(70).tolist()
        ]
    }
    response = client.post("/predict_batch/", json=sample_features)
    assert response.status_code == 200
    data = response.json()
    # Expect a unified "predictions" key with a list of predictions.
    predictions = data.get("predictions")
    assert isinstance(predictions, list)
    assert len(predictions) == 2
    # Optionally, check that each prediction contains both model outputs.
    for pred in predictions:
        assert "xgboost" in pred
        assert "nn" in pred
