# # # import requests

# # # from deployment.app import app

# # # API_URL = "http://127.0.0.1:8000/predict_batch/"

# # # def test_batch_prediction():
# # #     sample_features = {
# # #         "features": [
# # #             [0.5, 0.6, 0.7, 0.8],
# # #             [0.3, 0.4, 0.5, 0.6]
# # #         ]
# # #     }
# # #     response = requests.post(API_URL, json=sample_features)
# # #     assert response.status_code == 200
# # #     predictions = response.json()["predictions"]
# # #     assert len(predictions) == 2

# # import requests

# # API_URL = "http://127.0.0.1:8000/predict_batch/"

# # def test_batch_prediction():
# #     sample_features = {
# #         "features": [
# #             [0.5, 0.6, 0.7, 0.8],
# #             [0.3, 0.4, 0.5, 0.6]
# #         ]
# #     }
# #     response = requests.post(API_URL, json=sample_features)
# #     assert response.status_code == 200
# #     predictions = response.json()["predictions"]
# #     assert len(predictions) == 2

# from fastapi.testclient import TestClient
# from deployment.app import app

# client = TestClient(app)

# def test_batch_prediction():
#     sample_features = {
#         "features": [
#             [0.5, 0.6, 0.7, 0.8],
#             [0.3, 0.4, 0.5, 0.6]
#         ]
#     }
#     response = client.post("/predict_batch/", json=sample_features)
#     assert response.status_code == 200
#     predictions = response.json()["predictions"]
#     assert len(predictions) == 2

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
    predictions = response.json()["predictions"]
    assert isinstance(predictions, list)
    assert len(predictions) == 2
