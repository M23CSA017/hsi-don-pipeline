"""
evaluate.py

Model evaluation functions: cross-validation, metrics, result plotting, and SHAP analysis.
"""

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
import xgboost as xgb

def evaluate_model(model,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   random_state: int = 42) -> tuple:
    """
    Evaluate model using K-Fold cross-validation and test set performance.
    Returns a dictionary of metrics and test predictions.
    """
    logger = logging.getLogger(__name__)

    try:
        cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring='neg_mean_squared_error',
            n_jobs=4
        )
        rmse_scores = np.sqrt(-cv_scores)

        # Fit model on full training data
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            'CV_RMSE_mean': rmse_scores.mean(),
            'CV_RMSE_std': rmse_scores.std(),
            'Test_MAE': mean_absolute_error(y_test, y_pred),
            'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'Test_R2': r2_score(y_test, y_pred)
        }
        logger.info("Model evaluation completed.")
        return metrics, y_pred
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise e

def plot_results(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> None:
    """
    Create evaluation plots with scatter of actual vs. predicted,
    residuals, and error distribution.
    """
    plt.figure(figsize=(15, 5))

    # Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.6, label='Predictions')
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             'r--', label='Perfect Prediction')
    plt.xlabel('Actual DON (ppb)')
    plt.ylabel('Predicted DON (ppb)')
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Residuals
    residuals = y_true - y_pred
    plt.subplot(1, 3, 2)
    plt.scatter(y_pred, residuals, alpha=0.6, label='Residuals')
    plt.axhline(y=0, color='r', linestyle='--', label='Zero Error')
    plt.xlabel('Predicted Values (ppb)')
    plt.ylabel('Residuals (ppb)')
    plt.title(f'{model_name} - Residual Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Error Distribution
    plt.subplot(1, 3, 3)
    sns.histplot(residuals, kde=True, label='Error Distribution')
    plt.xlabel('Prediction Error (ppb)')
    plt.title(f'{model_name} - Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
def shap_analysis(model,
                  X_train: pd.DataFrame,
                  save_path: str = "data/models/shap_explainer.pkl") -> None:
    """
    Perform SHAP analysis on the trained model and produce summary plots.

    :param model: Trained XGBoost model
    :param X_train: Training features (without the target)
    :param save_path: Path to save the SHAP explainer model
    """
    import os
    import joblib
    logger = logging.getLogger(__name__)

    try:
        # Get absolute path to models directory
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Current file: evaluate.py
        models_dir = os.path.abspath(os.path.join(base_dir, "../../data/models/"))
        save_path = os.path.join(models_dir, "shap_explainer.pkl")

        # Ensure directory exists
        os.makedirs(models_dir, exist_ok=True)

        # Built-in feature importance
        plt.figure(figsize=(10, 8))
        xgb.plot_importance(model, max_num_features=20, height=0.8)
        plt.title('Top 20 Feature Importances\n(XGBoost Built-in)')
        plt.xlabel('Feature Importance Score')
        plt.tight_layout()
        plt.show()

        # SHAP Analysis
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_train)

        # SHAP Summary Plot - Bar
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=20, show=False)
        plt.title('SHAP Feature Importance\n(Mean Absolute Impact on Model Output)')
        plt.xlabel('Mean |SHAP Value| (Average Impact on Prediction)')
        plt.tight_layout()
        plt.show()

        # SHAP Summary Plot - Detailed
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X_train, max_display=10, show=False)
        plt.title('SHAP Value Distribution\n(Feature Impact on Predictions)')
        plt.xlabel('SHAP Value (Impact on Model Output in ppb)')
        plt.tight_layout()
        plt.show()

        # Save the SHAP explainer
        joblib.dump(explainer, save_path)
        logger.info(f"✅ SHAP explainer saved to {save_path}")
    except Exception as e:
        logger.error(f"❗ Error during SHAP analysis: {e}")
        raise e
