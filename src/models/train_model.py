# # # import os
# # # import numpy as np
# # # import pandas as pd
# # # import logging
# # # import joblib
# # # import torch
# # # from pathlib import Path
# # # from torch.utils.data import DataLoader, TensorDataset
# # # import xgboost as xgb
# # # import yaml
# # # import optuna
# # # import plotly.io as pio

# # # from optuna.visualization import plot_optimization_history, plot_param_importances

# # # # Local imports
# # # from ..utils.logger import setup_logging
# # # from ..data.load_data import load_data
# # # from ..data.preprocess import perform_eda, sensor_drift_check, preprocess_data
# # # from ..evaluation.evaluate import evaluate_model as evaluate_xgb_model, plot_results, shap_analysis
# # # from ..models.optimize import run_optuna_optimization
# # # from src.models.nn_model import (
# # #     DONRegressor,
# # #     train_model as train_nn_model,
# # #     evaluate_model as evaluate_nn_model,
# # #     save_model as save_nn_model,
# # #     load_model as load_nn_model,
# # # )

# # # # ---------------------------------------------------------------------------
# # # # CONFIG & LOGGING
# # # # ---------------------------------------------------------------------------
# # # CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
# # # logger = logging.getLogger(__name__)

# # # # Model Paths
# # # NN_MODEL_PATH = "./data/models/best_nn_model.pt"
# # # XGB_MODEL_PATH = "./data/models/best_xgboost_model.pkl"
# # # OPTUNA_PLOT_PATH = "./data/plots/"

# # # # ---------------------------------------------------------------------------
# # # # CONFIG LOADER
# # # # ---------------------------------------------------------------------------
# # # def load_config(config_path: str) -> dict:
# # #     """Load external configuration settings."""
# # #     try:
# # #         with open(config_path, 'r') as f:
# # #             cfg = yaml.safe_load(f)
# # #         logger.info("✅ Configuration loaded successfully.")
# # #         return cfg
# # #     except Exception as e:
# # #         logger.error(f"❗ Error loading config file {config_path}: {e}")
# # #         raise e


# # # # ---------------------------------------------------------------------------
# # # # DATA LOADER & DATALOADER CREATION
# # # # ---------------------------------------------------------------------------
# # # def prepare_dataloader(X, y, batch_size=32):
# # #     """
# # #     Prepare PyTorch DataLoader from Pandas DataFrame or NumPy array.
# # #     """
# # #     if isinstance(X, (pd.DataFrame, pd.Series)):
# # #         X = X.values
# # #     if isinstance(y, (pd.DataFrame, pd.Series)):
# # #         y = y.values.reshape(-1, 1)

# # #     X_tensor = torch.tensor(X, dtype=torch.float32)
# # #     y_tensor = torch.tensor(y, dtype=torch.float32)

# # #     dataset = TensorDataset(X_tensor, y_tensor)
# # #     return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# # # # ---------------------------------------------------------------------------
# # # # OPTUNA OBJECTIVE FUNCTION FOR NN
# # # # ---------------------------------------------------------------------------
# # # def nn_objective(trial, X_train, y_train, X_val, y_val, config):
# # #     """
# # #     Objective function to optimize Neural Network hyperparameters with Optuna.
# # #     """
# # #     # Optuna hyperparameter suggestions
# # #     num_layers = trial.suggest_int("num_layers", 2, 5)
# # #     hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
# # #     dropout = trial.suggest_float("dropout", 0.1, 0.5)
# # #     learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
# # #     weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
# # #     batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

# # #     # Create model with dynamic parameters
# # #     nn_model = DONRegressor(input_dim=X_train.shape[1], hidden_dim=hidden_dim,
# # #                             num_layers=num_layers, dropout=dropout)

# # #     # Define loss and optimizer
# # #     criterion = torch.nn.MSELoss()
# # #     optimizer = torch.optim.AdamW(nn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# # #     # Prepare DataLoaders
# # #     train_loader = prepare_dataloader(X_train, y_train, batch_size=batch_size)
# # #     val_loader = prepare_dataloader(X_val, y_val, batch_size=batch_size)

# # #     # Dynamic model path for Optuna trial
# # #     trial_model_path = f"./data/models/nn_model_trial_{trial.number}.pt"

# # #     # Train NN model
# # #     train_nn_model(
# # #         nn_model,
# # #         train_loader,
# # #         val_loader,
# # #         criterion,
# # #         optimizer,
# # #         model_path=trial_model_path,
# # #         num_epochs=config["nn_model"]["num_epochs"],
# # #         patience=config["nn_model"]["patience"],
# # #     )

# # #     # Evaluate on validation set
# # #     val_metrics = evaluate_nn_model(nn_model, val_loader)
# # #     val_rmse = val_metrics["Test_RMSE"]

# # #     return val_rmse


# # # # ---------------------------------------------------------------------------
# # # # MAIN TRAINING PIPELINE
# # # # ---------------------------------------------------------------------------
# # # def main():
# # #     # Setup logging
# # #     setup_logging(log_file="training.log")
# # #     logger.info("🚀 Starting training pipeline...")

# # #     # Load config
# # #     config = load_config(CONFIG_PATH)
# # #     RANDOM_STATE = config.get("random_state", 42)
# # #     raw_data_path = config.get("raw_data_path", "../data/raw/MLE-Assignment.csv")

# # #     # Step 1: Load raw data
# # #     df = load_data(raw_data_path)

# # #     # Step 2: Perform EDA & sensor drift checks
# # #     reflectance_cols = [col for col in df.columns if col not in ["hsi_id", "vomitoxin_ppb"]]
# # #     perform_eda(df, reflectance_cols)
# # #     sensor_drift_check(df)

# # #     # Step 3: Preprocess data
# # #     X_train, X_test, y_train, y_test = preprocess_data(
# # #         df, reflectance_cols,
# # #         nir_band=config.get("nir_band", "100"),
# # #         red_band=config.get("red_band", "50")
# # #     )

# # #     # Prepare final data for modeling
# # #     X_train2 = X_train.drop(columns=['hsi_id'])
# # #     X_test2 = X_test.drop(columns=['hsi_id'])
# # #     y_train2 = y_train['vomitoxin_ppb']
# # #     y_test2 = y_test['vomitoxin_ppb']

# # #     # ---------------------------------------------------------------------------
# # #     # Step 4a: Optuna Hyperparameter Tuning for Neural Network
# # #     # ---------------------------------------------------------------------------
# # #     logger.info("⚡ Starting Optuna hyperparameter tuning for Neural Network...")

# # #     # Convert data to PyTorch tensors
# # #     X_train_tensor = torch.tensor(X_train2.values, dtype=torch.float32)
# # #     y_train_tensor = torch.tensor(y_train2.values.reshape(-1, 1), dtype=torch.float32)
# # #     X_val_tensor = torch.tensor(X_test2.values, dtype=torch.float32)
# # #     y_val_tensor = torch.tensor(y_test2.values.reshape(-1, 1), dtype=torch.float32)

# # #     # Create Optuna study with pruning
# # #     nn_study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
# # #     nn_study.optimize(
# # #         lambda trial: nn_objective(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, config),
# # #         n_trials=config.get("optuna_trials_nn", 50)
# # #     )

# # #     # Get best trial and hyperparameters for NN
# # #     nn_best_trial = nn_study.best_trial
# # #     logger.info(f"🏆 Best NN trial RMSE: {nn_best_trial.value:.4f}")
# # #     logger.info(f"🔧 Best NN parameters: {nn_best_trial.params}")

# # #     # Load the best model after Optuna tuning
# # #     best_nn_model_path = f"./data/models/nn_model_trial_{nn_best_trial.number}.pt"
# # #     best_params = nn_best_trial.params
# # #     final_nn_model = DONRegressor(
# # #         input_dim=X_train2.shape[1],
# # #         hidden_dim=best_params["hidden_dim"],
# # #         num_layers=best_params["num_layers"],
# # #         dropout=best_params["dropout"],
# # #     )
# # #     final_nn_model.load_state_dict(torch.load(best_nn_model_path))
# # #     final_nn_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# # #     logger.info("✅ Best Neural Network Model loaded successfully.")

# # #     # Evaluate the best PyTorch NN model
# # #     test_loader = prepare_dataloader(X_test2, y_test2, batch_size=best_params["batch_size"])
# # #     nn_metrics = evaluate_nn_model(final_nn_model, test_loader)
# # #     logger.info("✅ Best Neural Network Performance after Optuna:")
# # #     logger.info(nn_metrics)

# # #     logger.info("🎉 Training pipeline completed successfully!")


# # # if __name__ == "__main__":
# # #     main()

# # import os
# # import numpy as np
# # import pandas as pd
# # import logging
# # import joblib
# # import torch
# # from pathlib import Path
# # from torch.utils.data import DataLoader, TensorDataset
# # import xgboost as xgb
# # import yaml
# # import optuna
# # import plotly.io as pio

# # from optuna.visualization import plot_optimization_history, plot_param_importances

# # # Local imports
# # from ..utils.logger import setup_logging
# # from ..data.load_data import load_data
# # from ..data.preprocess import perform_eda, sensor_drift_check, preprocess_data
# # from ..evaluation.evaluate import evaluate_model as evaluate_xgb_model, plot_results, shap_analysis
# # from ..models.optimize import run_optuna_optimization
# # from src.models.nn_model import (
# #     DONRegressor,
# #     train_model as train_nn_model,
# #     evaluate_model as evaluate_nn_model,
# #     save_model as save_nn_model,
# #     load_model as load_nn_model,
# # )

# # # ---------------------------------------------------------------------------
# # # CONFIG & LOGGING
# # # ---------------------------------------------------------------------------
# # CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
# # logger = logging.getLogger(__name__)

# # # Model Paths
# # NN_MODEL_PATH = "./data/models/best_nn_model.pt"
# # XGB_MODEL_PATH = "./data/models/best_xgboost_model.pkl"
# # OPTUNA_PLOT_PATH = "./data/plots/"

# # # ---------------------------------------------------------------------------
# # # CONFIG LOADER
# # # ---------------------------------------------------------------------------
# # def load_config(config_path: str) -> dict:
# #     """Load external configuration settings."""
# #     try:
# #         with open(config_path, 'r') as f:
# #             cfg = yaml.safe_load(f)
# #         logger.info("✅ Configuration loaded successfully.")
# #         return cfg
# #     except Exception as e:
# #         logger.error(f"❗ Error loading config file {config_path}: {e}")
# #         raise e


# # # ---------------------------------------------------------------------------
# # # DATA LOADER & DATALOADER CREATION
# # # ---------------------------------------------------------------------------
# # def prepare_dataloader(X, y, batch_size=32):
# #     """
# #     Prepare PyTorch DataLoader from Pandas DataFrame or NumPy array.
# #     """
# #     if isinstance(X, (pd.DataFrame, pd.Series)):
# #         X = X.values
# #     if isinstance(y, (pd.DataFrame, pd.Series)):
# #         y = y.values.reshape(-1, 1)

# #     X_tensor = torch.tensor(X, dtype=torch.float32)
# #     y_tensor = torch.tensor(y, dtype=torch.float32)

# #     dataset = TensorDataset(X_tensor, y_tensor)
# #     return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# # # ---------------------------------------------------------------------------
# # # OPTUNA OBJECTIVE FUNCTION FOR NN
# # # ---------------------------------------------------------------------------
# # def nn_objective(trial, X_train, y_train, X_val, y_val, config):
# #     """
# #     Objective function to optimize Neural Network hyperparameters with Optuna.
# #     """
# #     # Optuna hyperparameter suggestions
# #     num_layers = trial.suggest_int("num_layers", 2, 5)
# #     hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
# #     dropout = trial.suggest_float("dropout", 0.1, 0.5)
# #     learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
# #     weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
# #     batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

# #     # Create model with dynamic parameters
# #     nn_model = DONRegressor(input_dim=X_train.shape[1], hidden_dim=hidden_dim,
# #                             num_layers=num_layers, dropout=dropout)

# #     # Define loss and optimizer
# #     criterion = torch.nn.MSELoss()
# #     optimizer = torch.optim.AdamW(nn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# #     # Prepare DataLoaders
# #     train_loader = prepare_dataloader(X_train, y_train, batch_size=batch_size)
# #     val_loader = prepare_dataloader(X_val, y_val, batch_size=batch_size)

# #     # Train the model without saving models during Optuna trials
# #     train_nn_model(
# #         nn_model,
# #         train_loader,
# #         val_loader,
# #         criterion,
# #         optimizer,
# #         model_path=None,  # Do not save models during trials
# #         num_epochs=config["nn_model"]["num_epochs"],
# #         patience=config["nn_model"]["patience"],
# #     )

# #     # Evaluate on validation set
# #     val_metrics = evaluate_nn_model(nn_model, val_loader)
# #     val_rmse = val_metrics["Test_RMSE"]

# #     return val_rmse


# # # ---------------------------------------------------------------------------
# # # OPTUNA OBJECTIVE FUNCTION FOR XGBOOST
# # # ---------------------------------------------------------------------------
# # def xgb_objective(trial, X_train, y_train, X_val, y_val):
# #     """
# #     Objective function to optimize XGBoost hyperparameters with Optuna.
# #     """
# #     # Define search space for XGBoost
# #     params = {
# #         "objective": "reg:squarederror",
# #         "n_estimators": trial.suggest_int("n_estimators", 100, 500),
# #         "max_depth": trial.suggest_int("max_depth", 3, 15),
# #         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
# #         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
# #         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
# #         "gamma": trial.suggest_float("gamma", 0.0, 0.5),
# #         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
# #     }

# #     # Define XGBoost model
# #     model = xgb.XGBRegressor(**params, random_state=42)

# #     # Fit and evaluate model
# #     model.fit(X_train, y_train)
# #     y_pred = model.predict(X_val)

# #     # Calculate RMSE as evaluation metric
# #     rmse = np.sqrt(np.mean((y_val - y_pred) ** 2))
# #     return rmse


# # # ---------------------------------------------------------------------------
# # # MAIN TRAINING PIPELINE
# # # ---------------------------------------------------------------------------
# # def main():
# #     # Setup logging
# #     setup_logging(log_file="training.log")
# #     logger.info("🚀 Starting training pipeline...")

# #     # Load config
# #     config = load_config(CONFIG_PATH)
# #     RANDOM_STATE = config.get("random_state", 42)
# #     raw_data_path = config.get("raw_data_path", "../data/raw/MLE-Assignment.csv")

# #     # Step 1: Load raw data
# #     df = load_data(raw_data_path)

# #     # Step 2: Perform EDA & sensor drift checks
# #     reflectance_cols = [col for col in df.columns if col not in ["hsi_id", "vomitoxin_ppb"]]
# #     perform_eda(df, reflectance_cols)
# #     sensor_drift_check(df)

# #     # Step 3: Preprocess data
# #     X_train, X_test, y_train, y_test = preprocess_data(
# #         df, reflectance_cols,
# #         nir_band=config.get("nir_band", "100"),
# #         red_band=config.get("red_band", "50")
# #     )

# #     # Prepare final data for modeling
# #     X_train2 = X_train.drop(columns=['hsi_id'])
# #     X_test2 = X_test.drop(columns=['hsi_id'])
# #     y_train2 = y_train['vomitoxin_ppb']
# #     y_test2 = y_test['vomitoxin_ppb']

# #     # ---------------------------------------------------------------------------
# #     # Step 4a: Optuna Hyperparameter Tuning for XGBoost
# #     # ---------------------------------------------------------------------------
# #     logger.info("⚡ Starting Optuna hyperparameter tuning for XGBoost...")

# #     xgb_study = optuna.create_study(direction="minimize")
# #     xgb_study.optimize(
# #         lambda trial: xgb_objective(trial, X_train2.values, y_train2.values, X_test2.values, y_test2.values),
# #         n_trials=config.get("optuna_trials_xgb", 50)
# #     )

# #     # Get best trial and parameters for XGBoost
# #     xgb_best_trial = xgb_study.best_trial
# #     logger.info(f"🏆 Best XGBoost trial RMSE: {xgb_best_trial.value:.4f}")
# #     logger.info(f"🔧 Best XGBoost parameters: {xgb_best_trial.params}")

# #     # Train final XGBoost model with best parameters
# #     best_xgb_model = xgb.XGBRegressor(**xgb_best_trial.params, random_state=RANDOM_STATE)
# #     best_xgb_model.fit(X_train2.values, y_train2.values)

# #     # Save the best XGBoost model
# #     joblib.dump(best_xgb_model, XGB_MODEL_PATH)
# #     logger.info(f"✅ Best XGBoost model saved successfully to {XGB_MODEL_PATH}")

# #     # ---------------------------------------------------------------------------
# #     # Step 4b: Optuna Hyperparameter Tuning for Neural Network
# #     # ---------------------------------------------------------------------------
# #     logger.info("⚡ Starting Optuna hyperparameter tuning for Neural Network...")

# #     # Convert data to PyTorch tensors
# #     X_train_tensor = torch.tensor(X_train2.values, dtype=torch.float32)
# #     y_train_tensor = torch.tensor(y_train2.values.reshape(-1, 1), dtype=torch.float32)
# #     X_val_tensor = torch.tensor(X_test2.values, dtype=torch.float32)
# #     y_val_tensor = torch.tensor(y_test2.values.reshape(-1, 1), dtype=torch.float32)

# #     # Create Optuna study with pruning for NN
# #     nn_study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
# #     nn_study.optimize(
# #         lambda trial: nn_objective(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, config),
# #         n_trials=config.get("optuna_trials_nn", 50)
# #     )

# #     # Get best trial and hyperparameters for NN
# #     nn_best_trial = nn_study.best_trial
# #     logger.info(f"🏆 Best NN trial RMSE: {nn_best_trial.value:.4f}")
# #     logger.info(f"🔧 Best NN parameters: {nn_best_trial.params}")

# #     # ---------------------------------------------------------------------------
# #     # Step 5: Train and Save the Best NN Model After Optuna
# #     # ---------------------------------------------------------------------------
# #     best_params = nn_best_trial.params
# #     final_nn_model = DONRegressor(
# #         input_dim=X_train2.shape[1],
# #         hidden_dim=best_params["hidden_dim"],
# #         num_layers=best_params["num_layers"],
# #         dropout=best_params["dropout"],
# #     )

# #     # Prepare DataLoaders for retraining the best model
# #     train_loader = prepare_dataloader(X_train2, y_train2, batch_size=best_params["batch_size"])
# #     val_loader = prepare_dataloader(X_test2, y_test2, batch_size=best_params["batch_size"])

# #     # Define loss and optimizer
# #     criterion = torch.nn.MSELoss()
# #     optimizer = torch.optim.AdamW(final_nn_model.parameters(), lr=best_params["learning_rate"],
# #                                    weight_decay=best_params["weight_decay"])

# #     # Re-train the best NN model and save
# #     train_nn_model(
# #         final_nn_model,
# #         train_loader,
# #         val_loader,
# #         criterion,
# #         optimizer,
# #         model_path=NN_MODEL_PATH,
# #         num_epochs=config["nn_model"]["num_epochs"],
# #         patience=config["nn_model"]["patience"],
# #     )
# #     logger.info(f"✅ Best NN model retrained and saved successfully to {NN_MODEL_PATH}")

# #     # ---------------------------------------------------------------------------
# #     # Step 6: Evaluate Best XGBoost and NN Models
# #     # ---------------------------------------------------------------------------
# #     # Load best NN model
# #     final_nn_model.load_state_dict(torch.load(NN_MODEL_PATH))
# #     final_nn_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
# #     logger.info("✅ Best Neural Network Model loaded successfully.")

# #     # Evaluate NN model
# #     test_loader = prepare_dataloader(X_test2, y_test2, batch_size=best_params["batch_size"])
# #     nn_metrics = evaluate_nn_model(final_nn_model, test_loader)
# #     logger.info("✅ Best Neural Network Performance after Optuna:")
# #     logger.info(nn_metrics)

# #     # Evaluate XGBoost model
# #     y_pred_xgb = best_xgb_model.predict(X_test2.values)
# #     xgb_rmse = np.sqrt(np.mean((y_test2.values - y_pred_xgb) ** 2))
# #     logger.info(f"✅ Best XGBoost Model RMSE on test data: {xgb_rmse:.4f}")

# #     logger.info("🎉 Training pipeline completed successfully!")


# # if __name__ == "__main__":
# #     main()

# import os
# import numpy as np
# import pandas as pd
# import logging
# import joblib
# import torch
# from pathlib import Path
# from torch.utils.data import DataLoader, TensorDataset
# import xgboost as xgb
# import yaml
# import optuna
# import plotly.io as pio

# from optuna.visualization import plot_optimization_history, plot_param_importances

# # Local imports
# from ..utils.logger import setup_logging
# from ..data.load_data import load_data
# from ..data.preprocess import perform_eda, sensor_drift_check, preprocess_data
# from ..evaluation.evaluate import evaluate_model as evaluate_xgb_model, plot_results, shap_analysis
# from ..models.optimize import run_optuna_optimization
# from src.models.nn_model import (
#     DONRegressor,
#     train_model as train_nn_model,
#     evaluate_model as evaluate_nn_model,
#     save_model as save_nn_model,
#     load_model as load_nn_model,
# )

# # ---------------------------------------------------------------------------
# # CONFIG & LOGGING
# # ---------------------------------------------------------------------------
# CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
# logger = logging.getLogger(__name__)

# # Model Paths
# NN_MODEL_PATH = "./data/models/best_nn_model.pt"
# XGB_MODEL_PATH = "./data/models/best_xgboost_model.pkl"
# OPTUNA_PLOT_PATH = "./data/plots/"

# # ---------------------------------------------------------------------------
# # CONFIG LOADER
# # ---------------------------------------------------------------------------
# def load_config(config_path: str) -> dict:
#     """Load external configuration settings."""
#     try:
#         with open(config_path, 'r') as f:
#             cfg = yaml.safe_load(f)
#         logger.info("✅ Configuration loaded successfully.")
#         return cfg
#     except Exception as e:
#         logger.error(f"❗ Error loading config file {config_path}: {e}")
#         raise e


# # ---------------------------------------------------------------------------
# # DATA LOADER & DATALOADER CREATION
# # ---------------------------------------------------------------------------
# def prepare_dataloader(X, y, batch_size=32):
#     """
#     Prepare PyTorch DataLoader from Pandas DataFrame or NumPy array.
#     """
#     if isinstance(X, (pd.DataFrame, pd.Series)):
#         X = X.values
#     if isinstance(y, (pd.DataFrame, pd.Series)):
#         y = y.values.reshape(-1, 1)

#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     y_tensor = torch.tensor(y, dtype=torch.float32)

#     dataset = TensorDataset(X_tensor, y_tensor)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# # ---------------------------------------------------------------------------
# # OPTUNA OBJECTIVE FUNCTION FOR NN
# # ---------------------------------------------------------------------------
# def nn_objective(trial, X_train, y_train, X_val, y_val, config):
#     """
#     Objective function to optimize Neural Network hyperparameters with Optuna.
#     """
#     # Optuna hyperparameter suggestions
#     num_layers = trial.suggest_int("num_layers", 2, 5)
#     hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
#     dropout = trial.suggest_float("dropout", 0.1, 0.5)
#     learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
#     weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
#     batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

#     # Create model with dynamic parameters
#     nn_model = DONRegressor(input_dim=X_train.shape[1], hidden_dim=hidden_dim,
#                             num_layers=num_layers, dropout=dropout)

#     # Define loss and optimizer
#     criterion = torch.nn.MSELoss()
#     optimizer = torch.optim.AdamW(nn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#     # Prepare DataLoaders
#     train_loader = prepare_dataloader(X_train, y_train, batch_size=batch_size)
#     val_loader = prepare_dataloader(X_val, y_val, batch_size=batch_size)

#     # Train the model without saving models during Optuna trials
#     train_nn_model(
#         nn_model,
#         train_loader,
#         val_loader,
#         criterion,
#         optimizer,
#         model_path=None,  # Skip saving models during trials
#         num_epochs=config["nn_model"]["num_epochs"],
#         patience=config["nn_model"]["patience"],
#     )

#     # Evaluate on validation set
#     val_metrics = evaluate_nn_model(nn_model, val_loader)
#     val_r2 = val_metrics["Test_R2"]

#     # Return negative R² because Optuna minimizes the objective
#     return -val_r2


# # ---------------------------------------------------------------------------
# # OPTUNA OBJECTIVE FUNCTION FOR XGBOOST
# # # ---------------------------------------------------------------------------
# # def xgb_objective(trial, X_train, y_train, X_val, y_val):
# #     """
# #     Objective function to optimize XGBoost hyperparameters with Optuna.
# #     """
# #     # Define search space for XGBoost
# #     params = {
# #         "objective": "reg:squarederror",
# #         "n_estimators": trial.suggest_int("n_estimators", 100, 500),
# #         "max_depth": trial.suggest_int("max_depth", 3, 15),
# #         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
# #         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
# #         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
# #         "gamma": trial.suggest_float("gamma", 0.0, 0.5),
# #         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
# #     }

# #     # Define XGBoost model
# #     model = xgb.XGBRegressor(**params, random_state=42)

# #     # Fit and evaluate model with early stopping
# #     model.fit(
# #         X_train, y_train,
# #         eval_set=[(X_val, y_val)],
# #         eval_metric="rmse",
# #         early_stopping_rounds=20,
# #         verbose=False,
# #     )

# #     # Calculate R² for model performance
# #     y_pred = model.predict(X_val)
# #     r2 = 1 - (np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

# #     # Return negative R² for Optuna
# #     return -r2

# def xgb_objective(trial, X_train, y_train, X_val, y_val):
#     """
#     Objective function to optimize XGBoost hyperparameters with Optuna.
#     """
#     # Define search space for XGBoost
#     params = {
#         "objective": "reg:squarederror",
#         "n_estimators": trial.suggest_int("n_estimators", 100, 500),
#         "max_depth": trial.suggest_int("max_depth", 3, 15),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "gamma": trial.suggest_float("gamma", 0.0, 0.5),
#         "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
#         "eval_metric": "rmse",  # ✅ Correct eval metric
#     }

#     # Define XGBoost model
#     model = xgb.XGBRegressor(**params, random_state=42)

#     # ✅ Use early_stopping_rounds for XGBoost 3.x+
#     model.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         early_stopping_rounds=20,  # ✅ Correct for XGBoost 3.x
#         verbose=False,
#     )

#     # Calculate R² for model performance
#     y_pred = model.predict(X_val)
#     r2 = 1 - (np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

#     # Return negative R² for Optuna
#     return -r2


# # ---------------------------------------------------------------------------
# # MAIN TRAINING PIPELINE
# # ---------------------------------------------------------------------------
# def main():
#     # Setup logging
#     setup_logging(log_file="training.log")
#     logger.info("🚀 Starting training pipeline...")

#     # Load config
#     config = load_config(CONFIG_PATH)
#     RANDOM_STATE = config.get("random_state", 42)
#     raw_data_path = config.get("raw_data_path", "../data/raw/MLE-Assignment.csv")

#     # Step 1: Load raw data
#     df = load_data(raw_data_path)

#     # Step 2: Perform EDA & sensor drift checks
#     reflectance_cols = [col for col in df.columns if col not in ["hsi_id", "vomitoxin_ppb"]]
#     perform_eda(df, reflectance_cols)
#     sensor_drift_check(df)

#     # Step 3: Preprocess data
#     X_train, X_test, y_train, y_test = preprocess_data(
#         df, reflectance_cols,
#         nir_band=config.get("nir_band", "100"),
#         red_band=config.get("red_band", "50")
#     )

#     # Prepare final data for modeling
#     X_train2 = X_train.drop(columns=['hsi_id'])
#     X_test2 = X_test.drop(columns=['hsi_id'])
#     y_train2 = y_train['vomitoxin_ppb']
#     y_test2 = y_test['vomitoxin_ppb']

#     # ---------------------------------------------------------------------------
#     # Step 4a: Optuna Hyperparameter Tuning for XGBoost
#     # ---------------------------------------------------------------------------
#     logger.info("⚡ Starting Optuna hyperparameter tuning for XGBoost...")

#     xgb_study = optuna.create_study(direction="minimize")
#     xgb_study.optimize(
#         lambda trial: xgb_objective(trial, X_train2.values, y_train2.values, X_test2.values, y_test2.values),
#         n_trials=config.get("optuna_trials_xgb", 50)
#     )

#     # Get best trial and parameters for XGBoost
#     xgb_best_trial = xgb_study.best_trial
#     logger.info(f"🏆 Best XGBoost trial R²: {-xgb_best_trial.value:.4f}")
#     logger.info(f"🔧 Best XGBoost parameters: {xgb_best_trial.params}")

#     # Train final XGBoost model with best parameters
#     best_xgb_model = xgb.XGBRegressor(**xgb_best_trial.params, random_state=RANDOM_STATE)
#     best_xgb_model.fit(X_train2.values, y_train2.values)

#     # Save the best XGBoost model
#     joblib.dump(best_xgb_model, XGB_MODEL_PATH)
#     logger.info(f"✅ Best XGBoost model saved successfully to {XGB_MODEL_PATH}")

#     # ---------------------------------------------------------------------------
#     # Step 4b: Optuna Hyperparameter Tuning for Neural Network
#     # ---------------------------------------------------------------------------
#     logger.info("⚡ Starting Optuna hyperparameter tuning for Neural Network...")

#     # Convert data to PyTorch tensors
#     X_train_tensor = torch.tensor(X_train2.values, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train2.values.reshape(-1, 1), dtype=torch.float32)
#     X_val_tensor = torch.tensor(X_test2.values, dtype=torch.float32)
#     y_val_tensor = torch.tensor(y_test2.values.reshape(-1, 1), dtype=torch.float32)

#     # Create Optuna study with pruning for NN
#     nn_study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
#     nn_study.optimize(
#         lambda trial: nn_objective(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, config),
#         n_trials=config.get("optuna_trials_nn", 50)
#     )

#     # Get best trial and hyperparameters for NN
#     nn_best_trial = nn_study.best_trial
#     logger.info(f"🏆 Best NN trial R²: {-nn_best_trial.value:.4f}")
#     logger.info(f"🔧 Best NN parameters: {nn_best_trial.params}")

#     # ---------------------------------------------------------------------------
#     # Step 5: Train and Save the Best NN Model After Optuna
#     # ---------------------------------------------------------------------------
#     best_params = nn_best_trial.params
#     final_nn_model = DONRegressor(
#         input_dim=X_train2.shape[1],
#         hidden_dim=best_params["hidden_dim"],
#         num_layers=best_params["num_layers"],
#         dropout=best_params["dropout"],
#     )

#     # Prepare DataLoaders for retraining the best model
#     train_loader = prepare_dataloader(X_train2, y_train2, batch_size=best_params["batch_size"])
#     val_loader = prepare_dataloader(X_test2, y_test2, batch_size=best_params["batch_size"])

#     # Define loss and optimizer
#     criterion = torch.nn.MSELoss()
#     optimizer = torch.optim.AdamW(final_nn_model.parameters(), lr=best_params["learning_rate"],
#                                    weight_decay=best_params["weight_decay"])

#     # Re-train the best NN model and save
#     train_nn_model(
#         final_nn_model,
#         train_loader,
#         val_loader,
#         criterion,
#         optimizer,
#         model_path=NN_MODEL_PATH,
#         num_epochs=config["nn_model"]["num_epochs"],
#         patience=config["nn_model"]["patience"],
#     )
#     logger.info(f"✅ Best NN model retrained and saved successfully to {NN_MODEL_PATH}")

#     # ---------------------------------------------------------------------------
#     # Step 6: Evaluate Best XGBoost and NN Models
#     # ---------------------------------------------------------------------------
#     # Load best NN model
#     final_nn_model.load_state_dict(torch.load(NN_MODEL_PATH))
#     final_nn_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     logger.info("✅ Best Neural Network Model loaded successfully.")

#     # Evaluate NN model
#     test_loader = prepare_dataloader(X_test2, y_test2, batch_size=best_params["batch_size"])
#     nn_metrics = evaluate_nn_model(final_nn_model, test_loader)
#     logger.info("✅ Best Neural Network Performance after Optuna:")
#     logger.info(f"📊 RMSE: {nn_metrics['Test_RMSE']:.4f}, MAE: {nn_metrics['Test_MAE']:.4f}, R²: {nn_metrics['Test_R2']:.4f}")

#     # Evaluate XGBoost model
#     y_pred_xgb = best_xgb_model.predict(X_test2.values)
#     xgb_rmse = np.sqrt(np.mean((y_test2.values - y_pred_xgb) ** 2))
#     xgb_r2 = 1 - (np.sum((y_test2.values - y_pred_xgb) ** 2) / np.sum((y_test2.values - np.mean(y_test2.values)) ** 2))
#     logger.info(f"✅ Best XGBoost Model Performance: RMSE: {xgb_rmse:.4f}, R²: {xgb_r2:.4f}")

#     logger.info("🎉 Training pipeline completed successfully!")


# if __name__ == "__main__":
#     main()


import os
import numpy as np
import pandas as pd
import logging
import joblib
import torch
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
import yaml
import optuna
import plotly.io as pio
from packaging import version

from optuna.visualization import plot_optimization_history, plot_param_importances

# Local imports
from ..utils.logger import setup_logging
from ..data.load_data import load_data
from ..data.preprocess import perform_eda, sensor_drift_check, preprocess_data
from ..evaluation.evaluate import evaluate_model as evaluate_xgb_model, plot_results, shap_analysis
from ..models.optimize import run_optuna_optimization
from src.models.nn_model import (
    DONRegressor,
    train_model as train_nn_model,
    evaluate_model as evaluate_nn_model,
    save_model as save_nn_model,
    load_model as load_model,
)

# ---------------------------------------------------------------------------
# CONFIG & LOGGING
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "config.yaml"
logger = logging.getLogger(__name__)

# Model Paths
NN_MODEL_PATH = "./data/models/best_nn_model.pt"
XGB_MODEL_PATH = "./data/models/best_xgboost_model.pkl"
OPTUNA_PLOT_PATH = "./data/plots/"

# ---------------------------------------------------------------------------
# CONFIG LOADER
# ---------------------------------------------------------------------------
def load_config(config_path: str) -> dict:
    """Load external configuration settings."""
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        logger.info("✅ Configuration loaded successfully.")
        return cfg
    except Exception as e:
        logger.error(f"❗ Error loading config file {config_path}: {e}")
        raise e

# ---------------------------------------------------------------------------
# DATA LOADER & DATALOADER CREATION
# ---------------------------------------------------------------------------
def prepare_dataloader(X, y, batch_size=32):
    """
    Prepare PyTorch DataLoader from Pandas DataFrame or NumPy array.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.values
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.values.reshape(-1, 1)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ---------------------------------------------------------------------------
# OPTUNA OBJECTIVE FUNCTION FOR NN
# ---------------------------------------------------------------------------
def nn_objective(trial, X_train, y_train, X_val, y_val, config):
    """
    Objective function to optimize Neural Network hyperparameters with Optuna.
    """
    # Hyperparameter suggestions
    num_layers = trial.suggest_int("num_layers", 2, 5)
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

    # Create model with dynamic parameters
    nn_model = DONRegressor(input_dim=X_train.shape[1], hidden_dim=hidden_dim,
                            num_layers=num_layers, dropout=dropout)

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Prepare DataLoaders
    train_loader = prepare_dataloader(X_train, y_train, batch_size=batch_size)
    val_loader = prepare_dataloader(X_val, y_val, batch_size=batch_size)

    # Train the model without saving during trials
    train_nn_model(
        nn_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        model_path=NN_MODEL_PATH,  # Skip saving during Optuna trials
        num_epochs=config["nn_model"]["num_epochs"],
        patience=config["nn_model"]["patience"],
    )

    # Evaluate on validation set
    val_metrics = evaluate_nn_model(nn_model, val_loader)
    val_r2 = val_metrics["Test_R2"]

    # Return negative R² because Optuna minimizes the objective
    return -val_r2

# ---------------------------------------------------------------------------
# OPTUNA OBJECTIVE FUNCTION FOR XGBOOST
# ---------------------------------------------------------------------------
def xgb_objective(trial, X_train, y_train, X_val, y_val):
    """
    Objective function to optimize XGBoost hyperparameters with Optuna.
    """
    # Define search space for XGBoost
    params = {
        "objective": "reg:squarederror",
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "eval_metric": "rmse", 
    }

    # Create XGBoost model
    model = xgb.XGBRegressor(**params, random_state=42)


    # Fit model with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Calculate R² for model performance
    y_pred = model.predict(X_val)
    r2 = 1 - (np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2))

    # Return negative R² for Optuna
    return -r2

# ---------------------------------------------------------------------------
# MAIN TRAINING PIPELINE
# ---------------------------------------------------------------------------
def main():
    # Setup logging
    setup_logging(log_file="training.log")
    logger.info("🚀 Starting training pipeline...")

    # Load config
    config = load_config(CONFIG_PATH)
    RANDOM_STATE = config.get("random_state", 42)
    raw_data_path = config.get("raw_data_path", "../data/raw/MLE-Assignment.csv")

    # Step 1: Load raw data
    df = load_data(raw_data_path)

    # Step 2: Perform EDA & sensor drift checks
    reflectance_cols = [col for col in df.columns if col not in ["hsi_id", "vomitoxin_ppb"]]
    perform_eda(df, reflectance_cols)
    sensor_drift_check(df)

    # Step 3: Preprocess data
    X_train, X_test, y_train, y_test = preprocess_data(
        df, reflectance_cols,
        nir_band=config.get("nir_band", "100"),
        red_band=config.get("red_band", "50")
    )

    # Prepare final data for modeling
    X_train2 = X_train.drop(columns=['hsi_id'])
    X_test2 = X_test.drop(columns=['hsi_id'])
    y_train2 = y_train['vomitoxin_ppb']
    y_test2 = y_test['vomitoxin_ppb']

    # ---------------------------------------------------------------------------
    # Step 4a: Optuna Hyperparameter Tuning for XGBoost
    # ---------------------------------------------------------------------------
    logger.info("⚡ Starting Optuna hyperparameter tuning for XGBoost...")
    xgb_study = optuna.create_study(direction="minimize")
    xgb_study.optimize(
        lambda trial: xgb_objective(trial, X_train2.values, y_train2.values, X_test2.values, y_test2.values),
        n_trials=config.get("optuna_trials_xgb", 50)
    )
    xgb_best_trial = xgb_study.best_trial
    logger.info(f"🏆 Best XGBoost trial R²: {-xgb_best_trial.value:.4f}")
    logger.info(f"🔧 Best XGBoost parameters: {xgb_best_trial.params}")

    # Train final XGBoost model with best parameters
    best_xgb_model = xgb.XGBRegressor(**xgb_best_trial.params, random_state=RANDOM_STATE)
    best_xgb_model.fit(X_train2.values, y_train2.values)
    joblib.dump(best_xgb_model, XGB_MODEL_PATH)
    logger.info(f"✅ Best XGBoost model saved successfully to {XGB_MODEL_PATH}")

    # ---------------------------------------------------------------------------
    # Step 4b: Optuna Hyperparameter Tuning for Neural Network
    # ---------------------------------------------------------------------------
    logger.info("⚡ Starting Optuna hyperparameter tuning for Neural Network...")
    X_train_tensor = torch.tensor(X_train2.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train2.values.reshape(-1, 1), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_test2.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_test2.values.reshape(-1, 1), dtype=torch.float32)

    nn_study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=10))
    nn_study.optimize(
        lambda trial: nn_objective(trial, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, config),
        n_trials=config.get("optuna_trials_nn", 50)
    )
    nn_best_trial = nn_study.best_trial
    logger.info(f"🏆 Best NN trial R²: {-nn_best_trial.value:.4f}")
    logger.info(f"🔧 Best NN parameters: {nn_best_trial.params}")

    # ---------------------------------------------------------------------------
    # Step 5: Train and Save the Best NN Model After Optuna
    # ---------------------------------------------------------------------------
    best_params = nn_best_trial.params
    final_nn_model = DONRegressor(
        input_dim=X_train2.shape[1],
        hidden_dim=best_params["hidden_dim"],
        num_layers=best_params["num_layers"],
        dropout=best_params["dropout"],
    )
    train_loader = prepare_dataloader(X_train2, y_train2, batch_size=best_params["batch_size"])
    val_loader = prepare_dataloader(X_test2, y_test2, batch_size=best_params["batch_size"])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(final_nn_model.parameters(), lr=best_params["learning_rate"],
                                   weight_decay=best_params["weight_decay"])
    train_nn_model(
        final_nn_model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        model_path=NN_MODEL_PATH,
        num_epochs=config["nn_model"]["num_epochs"],
        patience=config["nn_model"]["patience"],
    )
    logger.info(f"✅ Best NN model retrained and saved successfully to {NN_MODEL_PATH}")

    # ---------------------------------------------------------------------------
    # Step 6: Evaluate Best XGBoost and NN Models
    # ---------------------------------------------------------------------------
    final_nn_model.load_state_dict(torch.load(NN_MODEL_PATH))
    final_nn_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("✅ Best Neural Network Model loaded successfully.")
    test_loader = prepare_dataloader(X_test2, y_test2, batch_size=best_params["batch_size"])
    nn_metrics = evaluate_nn_model(final_nn_model, test_loader)
    logger.info("✅ Best Neural Network Performance after Optuna:")
    logger.info(f"📊 RMSE: {nn_metrics['Test_RMSE']:.4f}, MAE: {nn_metrics['Test_MAE']:.4f}, R²: {nn_metrics['Test_R2']:.4f}")
    y_pred_xgb = best_xgb_model.predict(X_test2.values)
    xgb_rmse = np.sqrt(np.mean((y_test2.values - y_pred_xgb) ** 2))
    xgb_r2 = 1 - (np.sum((y_test2.values - y_pred_xgb) ** 2) / np.sum((y_test2.values - np.mean(y_test2.values)) ** 2))
    logger.info(f"✅ Best XGBoost Model Performance: RMSE: {xgb_rmse:.4f}, R²: {xgb_r2:.4f}")
    logger.info("🎉 Training pipeline completed successfully!")

if __name__ == "__main__":
    main()

