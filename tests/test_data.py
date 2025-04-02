import pandas as pd
import numpy as np
import pytest
from src.data.load_data import load_data
from src.data.preprocess import perform_eda, sensor_drift_check, preprocess_data

def test_load_data(tmp_path):
    # Create a temporary CSV file with dummy hyperspectral data (e.g., 100 bands)
    dummy_csv = tmp_path / "dummy_data.csv"
    n_bands = 100
    bands = {f"band{i}": np.random.rand(10) for i in range(n_bands)}
    dummy_data = pd.DataFrame({
        "hsi_id": range(10),
        **bands,
        "vomitoxin_ppb": np.random.rand(10) * 1000
    })
    dummy_data.to_csv(dummy_csv, index=False)
    
    data = load_data(str(dummy_csv))
    assert isinstance(data, pd.DataFrame)
    assert "vomitoxin_ppb" in data.columns

def test_preprocess_data(tmp_path):
    # Create a dummy DataFrame with enough samples and bands for PCA.
    n_samples = 100
    n_bands = 100
    bands = {f"band{i}": np.random.rand(n_samples) for i in range(n_bands)}
    data = pd.DataFrame({
        "hsi_id": range(n_samples),
        **bands,
        "vomitoxin_ppb": np.random.rand(n_samples) * 1000
    })
    reflectance_cols = [f"band{i}" for i in range(n_bands)]
    # Use arbitrary bands for nir and red that are in the data.
    X_train, X_test, y_train, y_test = preprocess_data(data, reflectance_cols, nir_band="band50", red_band="band25")
    
    # Check that the training data is not empty and no missing values in target.
    assert X_train.shape[0] > 0
    assert y_train.isnull().sum().sum() == 0
