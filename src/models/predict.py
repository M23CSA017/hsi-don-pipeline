# import torch
# import joblib
# import numpy as np
# import yaml
# from pathlib import Path
# from src.models.nn_model import DONRegressor

# # Model paths
# XGB_MODEL_PATH = "./data/models/best_xgboost_model.pkl"
# NN_MODEL_PATH = "./data/models/best_nn_model.pt"
# CONFIG_PATH = "./configs/config.yaml"


# # ------------------ Load Models ------------------

# def load_xgb_model(model_path=XGB_MODEL_PATH):
#     if not Path(model_path).exists():
#         raise FileNotFoundError(f"❌ XGBoost model not found at: {model_path}")
#     return joblib.load(model_path)


# def load_nn_model(model_path=NN_MODEL_PATH, config_path="./configs/config.yaml"):


#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)["nn_model"]

#     model = DONRegressor(
#         input_dim=config["input_dim"],
#         hidden_dim=config["hidden_dim"],
#         num_layers=config["num_layers"],
#         dropout=config["dropout"]
#     )

#     state_dict = torch.load(model_path, map_location=torch.device("cpu"))
#     model.load_state_dict(state_dict)
#     model.eval()
#     return model


# # ------------------ Predict with Both ------------------

# def predict_with_both_models(features, config_path=CONFIG_PATH):
#     """
#     Returns predictions from both XGBoost and NN models for a single input vector.
#     """
#     features = np.array(features)
#     if features.ndim == 1:
#         features = features.reshape(1, -1)

#     # XGBoost Prediction
#     xgb_model = load_xgb_model()
#     xgb_pred = xgb_model.predict(features)[0]

#     # NN Prediction
#     nn_model = load_nn_model(config_path=config_path)
#     tensor_input = torch.tensor(features, dtype=torch.float32)
#     with torch.no_grad():
#         nn_pred = nn_model(tensor_input).item()

#     return {
#         "xgboost_prediction": float(xgb_pred),
#         "nn_prediction": float(nn_pred)
#     }


# def predict_batch_with_both_models(features_list, config_path=CONFIG_PATH):
#     """
#     Returns predictions from both models for a batch of samples.
#     """
#     features_array = np.array(features_list)
#     xgb_model = load_xgb_model()
#     nn_model = load_nn_model(config_path=config_path)

#     # XGBoost predictions
#     xgb_preds = xgb_model.predict(features_array).tolist()

#     # NN predictions
#     tensor_input = torch.tensor(features_array, dtype=torch.float32)
#     with torch.no_grad():
#         nn_preds = nn_model(tensor_input).squeeze().tolist()

#     return {
#         "xgboost_predictions": xgb_preds,
#         "nn_predictions": nn_preds
#     }

import torch
import joblib
import numpy as np
import yaml
from pathlib import Path
from src.models.nn_model import DONRegressor

# Model paths
XGB_MODEL_PATH = "./data/models/best_xgboost_model.pkl"
NN_MODEL_PATH = "./data/models/best_nn_model.pt"
CONFIG_PATH = "./configs/config.yaml"


# ------------------ Load Models ------------------

def load_xgb_model(model_path=XGB_MODEL_PATH):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"❌ XGBoost model not found at: {model_path}")
    return joblib.load(model_path)


def load_nn_model(model_path=NN_MODEL_PATH, config_path=CONFIG_PATH):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["nn_model_inference"]

    model = DONRegressor(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"]
    )

    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


# ------------------ Predict with Both ------------------

def predict_with_both_models(features, config_path=CONFIG_PATH):
    """
    Returns predictions from both XGBoost and NN models for a single input vector.
    """
    features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(1, -1)

    # XGBoost Prediction
    xgb_model = load_xgb_model()
    xgb_pred = xgb_model.predict(features)[0]

    # NN Prediction
    nn_model = load_nn_model(config_path=config_path)
    tensor_input = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        nn_pred = nn_model(tensor_input).item()

    return {
        "xgboost_prediction": float(xgb_pred),
        "nn_prediction": float(nn_pred)
    }


def predict_batch_with_both_models(features_list, config_path=CONFIG_PATH):
    """
    Returns predictions from both models for a batch of samples.
    """
    features_array = np.array(features_list)

    # Load models
    xgb_model = load_xgb_model()
    nn_model = load_nn_model(config_path=config_path)

    # XGBoost predictions
    xgb_preds = xgb_model.predict(features_array).tolist()

    # NN predictions
    tensor_input = torch.tensor(features_array, dtype=torch.float32)
    with torch.no_grad():
        nn_preds = nn_model(tensor_input).squeeze().tolist()

    return {
        "xgboost_predictions": xgb_preds,
        "nn_predictions": nn_preds
    }
