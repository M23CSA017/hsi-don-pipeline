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
