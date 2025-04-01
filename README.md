# HSI DON Pipeline

## Overview
This repository implements a machine learning pipeline to predict mycotoxin levels (DON concentration) in corn samples using hyperspectral imaging data.

## Features
- **Data Loading & EDA**: Inspect raw CSV, check for anomalies, basic sensor drift placeholder.
- **Preprocessing**: Imputation, scaling, Z-score anomaly detection, PCA dimension reduction.
- **Model Training**: XGBoost baseline + hyperparameter tuning (Optuna).
- **Evaluation**: Cross-validation, standard regression metrics, SHAP interpretability.
- **Deployment**: Dockerfile and FastAPI service (`app.py`) for real-time predictions.
- **Testing**: Unit tests for data pipeline & model pipeline using `unittest`.

## Folder Structure
(See the detailed structure in the project.)

## Quick Start
1. **Clone the repo**:


## 🔍 Model Evaluation & Visualizations

- **Actual vs Predicted Plot** – Compares actual values with predicted values.
- **Residual Plot** – Analyzes model error and residual distribution.
- **SHAP Analysis** – Provides insights into feature importance using SHAP.

### 🎨 Visualization Outputs:
- XGBoost Evaluation Plots: `data/plots/xgb_results.png`
- Neural Network Evaluation Plots: `data/plots/nn_results.png`
- SHAP Feature Importance Plots:
    - `data/plots/xgb_feature_importance.png`
    - `data/plots/shap_bar_plot.png`
    - `data/plots/shap_summary_plot.png`

## 🚀 API Endpoints

- **POST /predict** – Predict DON concentration for a single sample.
- **POST /predict_batch/** – Predict DON concentration for multiple samples.

## 🐳 Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t don_prediction_api .

