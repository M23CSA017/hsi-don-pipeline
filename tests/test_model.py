
"""
test_model.py

Unit tests for model training and evaluation flows.
"""

import unittest
import os
import tempfile
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
import joblib

from src.models.predict import predict_don
from src.evaluation.evaluate import evaluate_model

# For testing the NN model, import DONRegressor and save_model
from src.models.nn_model import DONRegressor, save_model

class TestXGBoostPipeline(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test artifacts
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create dummy training and test data with sufficient rows and many bands
        n_train, n_test, n_features = 100, 20, 200  # 200 bands → PCA can extract 70 components
        band_names = [f"band{i}" for i in range(n_features)]
        train_df = pd.DataFrame(np.random.rand(n_train, n_features), columns=band_names)
        train_df["hsi_id"] = np.arange(1, n_train+1)
        y_train = pd.DataFrame({'vomitoxin_ppb': np.random.randint(0, 2000, size=n_train)})

        test_df = pd.DataFrame(np.random.rand(n_test, n_features), columns=band_names)
        test_df["hsi_id"] = np.arange(n_train+1, n_train+n_test+1)
        y_test = pd.DataFrame({'vomitoxin_ppb': np.random.randint(0, 2000, size=n_test)})

        # Save to CSV files in the temporary directory
        self.x_train_csv = os.path.join(self.temp_dir.name, "x_train.csv")
        self.x_test_csv = os.path.join(self.temp_dir.name, "x_test.csv")
        self.y_train_csv = os.path.join(self.temp_dir.name, "y_train.csv")
        self.y_test_csv = os.path.join(self.temp_dir.name, "y_test.csv")

        train_df.to_csv(self.x_train_csv, index=False)
        test_df.to_csv(self.x_test_csv, index=False)
        y_train.to_csv(self.y_train_csv, index=False)
        y_test.to_csv(self.y_test_csv, index=False)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_evaluate_model(self):
        # Load data from CSV files
        X_train = pd.read_csv(self.x_train_csv)
        X_test = pd.read_csv(self.x_test_csv)
        y_train = pd.read_csv(self.y_train_csv)['vomitoxin_ppb']
        y_test = pd.read_csv(self.y_test_csv)['vomitoxin_ppb']

        # Remove the 'hsi_id' column
        X_train = X_train.drop(columns=['hsi_id'])
        X_test = X_test.drop(columns=['hsi_id'])

        # Train a simple XGBoost model
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=10,
            max_depth=2,
            random_state=42
        )
        metrics, preds = evaluate_model(model, X_train, y_train, X_test, y_test, random_state=42)
        self.assertIn('Test_R2', metrics)
        self.assertTrue(isinstance(preds, np.ndarray))
        self.assertEqual(len(preds), len(y_test))

    def test_predict_don(self):
        # Load training data
        X_train = pd.read_csv(self.x_train_csv)
        y_train = pd.read_csv(self.y_train_csv)['vomitoxin_ppb']
        X_train_features = X_train.drop(columns=['hsi_id'])

        # -------------------------------
        # Test XGBoost Prediction
        # -------------------------------
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=10,
            max_depth=2,
            random_state=42
        )
        xgb_model.fit(X_train_features, y_train)
        xgb_model_path = os.path.join(self.temp_dir.name, 'best_xgboost_model.pkl')
        joblib.dump(xgb_model, xgb_model_path)

        sample_features = X_train_features.iloc[0].values
        xgb_prediction = predict_don(sample_features, model_path=xgb_model_path, use_nn=False)
        self.assertTrue(isinstance(xgb_prediction, float))

        # -------------------------------
        # Test Neural Network Prediction
        # -------------------------------
        nn_model = DONRegressor(
            input_dim=X_train_features.shape[1],
            hidden_dim=128,
            num_layers=3,
            dropout=0.2
        )
        nn_model_path = os.path.join(self.temp_dir.name, 'best_nn_model.pt')
        save_model(nn_model, nn_model_path)

        # Create a dummy YAML config file for the NN model parameters.
        config_path = os.path.join(self.temp_dir.name, 'model_config.yaml')
        dummy_config = {
            "nn_model_inference": { 
                "input_dim": X_train_features.shape[1],
                "hidden_dim": 128,
                "num_layers": 3,
                "dropout": 0.2,
                "num_epochs": 10,
                "patience": 10,
                "learning_rate": 0.001,
                "batch_size": 32
            },
            "optuna_trials": 5
        }
        with open(config_path, 'w') as f:
            yaml.dump(dummy_config, f)

        nn_prediction = predict_don(sample_features, model_path=nn_model_path, use_nn=True, config_path=config_path)
        self.assertTrue(isinstance(nn_prediction, float))

if __name__ == "__main__":
    unittest.main()
