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

from sklearn.metrics import r2_score
from hsi_don_pipeline.src.evaluation.evaluate import evaluate_model
from hsi_don_pipeline.src.models.predict import predict_don
import joblib

class TestXGBoostPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        # Create dummy training and test data
        n_train, n_test, n_features = 40, 10, 5
        # Minimal columns
        train_df = pd.DataFrame(np.random.rand(n_train, n_features), columns=[f'PC{i}' for i in range(1, n_features+1)])
        train_df["hsi_id"] = np.arange(1, n_train+1)
        y_train = pd.DataFrame({'vomitoxin_ppb': np.random.randint(0, 2000, size=n_train)})

        test_df = pd.DataFrame(np.random.rand(n_test, n_features), columns=[f'PC{i}' for i in range(1, n_features+1)])
        test_df["hsi_id"] = np.arange(n_train+1, n_train+n_test+1)
        y_test = pd.DataFrame({'vomitoxin_ppb': np.random.randint(0, 2000, size=n_test)})

        # Save to CSV
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
        # Load data
        X_train = pd.read_csv(self.x_train_csv)
        X_test = pd.read_csv(self.x_test_csv)
        y_train = pd.read_csv(self.y_train_csv)['vomitoxin_ppb']
        y_test = pd.read_csv(self.y_test_csv)['vomitoxin_ppb']

        # Remove ID
        X_train = X_train.drop(columns=['hsi_id'])
        X_test = X_test.drop(columns=['hsi_id'])

        # Simple model
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
        # Load data
        X_train = pd.read_csv(self.x_train_csv)
        y_train = pd.read_csv(self.y_train_csv)['vomitoxin_ppb']

        X_train2 = X_train.drop(columns=['hsi_id'])

        # Fit model & save
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=10,
            max_depth=2,
            random_state=42
        )
        model.fit(X_train2, y_train)

        model_path = os.path.join(self.temp_dir.name, 'best_xgboost_model.pkl')
        joblib.dump(model, model_path)

        sample_features = X_train2.iloc[0].values
        prediction = predict_don(sample_features, model_path=model_path)
        self.assertTrue(isinstance(prediction, float))

if __name__ == '__main__':
    unittest.main()
