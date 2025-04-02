# 🌽 HSI DON Pipeline

A production-ready machine learning pipeline for predicting **DON (Deoxynivalenol)** concentrations in corn using **Hyperspectral Imaging (HSI)** data. This pipeline is modular, interpretable, and deployable with Docker and FastAPI.

---

## 📆 Features

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
  - FastAPI-powered REST API with `/predict` and `/predict_batch`
  - Docker containerization using GitHub Actions
  - CI/CD integration with GitHub Workflows

- 🧚 **Testing & CI/CD**
  - Modular unit tests for each pipeline stage
  - Automatic linting and testing via **GitHub Actions**
  - Docker image built and pushed to **GitHub Container Registry**

---

## ⚙️ Project Structure

```
📁 hsi_don_pipeline/
👅 configs/                   # YAML-based configurations
📁 data/                      # Raw, processed data & saved models
📁 deployment/               # Dockerfile and FastAPI app
📁 notebooks/                # EDA and experiment notebooks
📁 src/
├── data/                 # Load & preprocess modules
├── evaluation/           # Evaluation, SHAP, residuals, plots
└── models/               # Training logic and model utils
📁 tests/                    # Unit tests
requirements.txt
requirements-dev.txt
README.md
```

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

## 🚀 FastAPI API Documentation

### 📌 Endpoints

#### `POST /predict`
- **Input**:
  ```json
  {
    "features": [float, float, ..., float]
  }
  ```
- **Output**:
  ```json
  {
    "xgboost_prediction": float,
    "nn_prediction": float
  }
  ```

#### `POST /predict_batch`
- **Input**:
  ```json
  {
    "features": [[float, float, ...], [float, float, ...], ...]
  }
  ```
- **Output**:
  ```json
  {
    "predictions": [
      {"xgboost": float, "nn": float},
      {"xgboost": float, "nn": float},
      ...
    ]
  }
  ```

---

## 🐳 Docker Deployment

### 🛠️ Build & Run Locally
```bash
# Build Docker image
$ docker build -t don-predictor -f deployment/Dockerfile .

# Run container
$ docker run -p 8000:8000 don-predictor
```

### 🌐 Access API
Go to: [http://127.0.0.1:8000](http://127.0.0.1:8000)

---

## 🔄 CI/CD (GitHub Actions)

### 1. **CI Pipeline (`.github/workflows/ci.yml`)**
- Runs on push/pull to `main`
- Installs dependencies
- Runs all unit tests with pytest
- Lints using flake8

### 2. **Docker Build (`.github/workflows/docker.yml`)**
- Automatically builds Docker image
- Pushes to GitHub Container Registry (GHCR) as `ghcr.io/<owner>/don-predictor:latest`

---

## 💡 Future Enhancements
- 🌟 Add out-of-distribution detection
- 🔊 Include logging with structured logs
- 🧠 Support model versioning (MLflow)
- ☁️ Deploy on AWS/GCP/Azure

---

## 👨‍💻 Author
Prabhat, M.Tech AI @ IIT Jodhpur

📬 Reach out for questions, collaborations, or feedback!
