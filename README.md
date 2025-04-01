# 🌽 HSI DON Pipeline

A production-ready machine learning pipeline for predicting **DON (Deoxynivalenol)** concentrations in corn using **Hyperspectral Imaging (HSI)** data. This pipeline is modular, interpretable, and deployable with Docker and FastAPI.

---

## 📦 Features

- ✅ **Data Ingestion & Exploration**
  - Missing value checks, duplicates, outlier detection
  - Sensor drift visualization (placeholder)
  - EDA visualizations (histograms, spectral plots, PCA, etc.)

- ⚙️ **Preprocessing**
  - Z-score based anomaly filtering
  - Imputation & standardization
  - NDVI-like index feature
  - PCA-based dimensionality reduction

- 🤖 **Model Training & Optimization**
  - XGBoost and Neural Network regressors
  - Hyperparameter tuning using **Optuna**
  - Cross-validation support
  - Residual analysis for error diagnostics

- 📊 **Evaluation & Explainability**
  - Regression metrics (MAE, RMSE, R²)
  - Residual & actual-vs-predicted plots
  - SHAP-based feature importance and bar plots
  - Model comparison summary plots

- 🚀 **Deployment**
  - FastAPI-powered REST API
  - Dockerfile for containerization
  - Real-time single & batch prediction endpoints

- 🧪 **Testing & CI/CD**
  - Modular unit tests for pipeline stages
  - GitHub Actions CI for every commit
  - Ready for GitHub Codespaces and local development

---


---

## 🔍 Model Evaluation & Visualizations

- **Actual vs Predicted Plot** – Compares true and predicted values.
- **Residual Plot** – Examines model prediction errors.
- **SHAP Analysis** – Explains model decisions using feature importance.

### 🎨 Visualization Outputs

- `data/plots/xgb_results.png` – XGBoost results
- `data/plots/nn_results.png` – Neural Network results
- `data/plots/shap_summary_plot.png` – SHAP Summary
- `data/plots/model_comparison.png` – Model comparison
- `data/plots/residuals_nn.png` – Residuals: Neural Network
- `data/plots/residuals_xgb.png` – Residuals: XGBoost

---

## 🚀 FastAPI Endpoints

- `POST /predict` – Predict DON concentration for a single sample
- `POST /predict_batch` – Predict DON concentrations for multiple samples

### Example usage:
```bash
curl -X POST http://127.0.0.1:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [0.12, 0.34, ..., 0.78]}'


