
"""
preprocess.py

Handles data exploration (EDA) and preprocessing steps such as
imputation, scaling, anomaly detection, and PCA dimensionality reduction.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple

# Define plot save directory using pathlib
BASE_DIR = Path(__file__).resolve().parent  # Get the current script's directory
PLOT_DIR = BASE_DIR / "outputs" / "plots" / "eda"

# Create directory if it doesn't exist
PLOT_DIR.mkdir(parents=True, exist_ok=True)
print(f"✅ PLOT_DIR is set to: {PLOT_DIR}")

logger = logging.getLogger(__name__)


def perform_eda(df: pd.DataFrame, reflectance_cols: list) -> None:
    """
    Perform EDA with visualizations and additional checks.
    Fixed version with proper numerical handling of wavelength bands.

    :param df: Raw DataFrame with reflectance columns.
    :param reflectance_cols: List of reflectance column names.
    """
    logger.info("\n📊 Summary Statistics for Reflectance Bands:")
    logger.info(str(df[reflectance_cols].describe().T.head()))

    logger.info("\n📊 Summary Statistics for Target (vomitoxin_ppb):")
    logger.info(str(df["vomitoxin_ppb"].describe()))

    # Check for missing values and duplicates
    missing_values = df.isnull().sum()
    logger.info("\n❗ Missing Values:")
    logger.info(str(missing_values[missing_values > 0]))
    logger.info(f"❗ Duplicate Rows: {df.duplicated().sum()}")

    # Convert band names to float for proper plotting
    band_numbers = np.array([float(band) for band in reflectance_cols])

    # 1. Average reflectance plot (fixed)
    plt.figure(figsize=(12, 6))
    plt.plot(band_numbers, df[reflectance_cols].mean(), label="Average Reflectance")
    plt.title("Average Reflectance Across Wavelengths")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.legend()
    plt.savefig(PLOT_DIR / "average_reflectance.png", bbox_inches="tight")
    plt.show()

    # 2. Distribution of DON concentration
    plt.figure(figsize=(8, 5))
    sns.histplot(df["vomitoxin_ppb"], kde=True)
    plt.title("Distribution of DON Concentration")
    plt.xlabel("DON (ppb)")
    plt.ylabel("Frequency")
    plt.savefig(PLOT_DIR / "don_distribution.png", bbox_inches="tight")
    plt.show()

    # 3. Histograms of selected bands (fixed)
    selected_bands = reflectance_cols[::50]
    for band in selected_bands[:3]:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[band], kde=True)
        plt.title(f"Distribution of Band {float(band):.1f} nm")
        plt.xlabel("Reflectance")
        plt.ylabel("Frequency")
        plt.savefig(PLOT_DIR / f"hist_band_{float(band):.1f}.png", bbox_inches="tight")
        plt.show()

    # 4. Boxplot of selected bands (fixed)
    plt.figure(figsize=(12, 6))
    boxplot_data = df[selected_bands].copy()
    boxplot_data.columns = [f"{float(col):.1f} nm" for col in boxplot_data.columns]
    sns.boxplot(data=boxplot_data)
    plt.title("Boxplot of Selected Reflectance Bands")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.xticks(rotation=45)
    plt.savefig(PLOT_DIR / "boxplot_selected_bands.png", bbox_inches="tight")
    plt.show()

    # 5. Correlation heatmap (fixed)
    plt.figure(figsize=(10, 8))
    corr_data = df[selected_bands].copy()
    corr_data.columns = [f"{float(col):.1f}" for col in corr_data.columns]
    sns.heatmap(corr_data.corr(), cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap of Selected Bands")
    plt.savefig(PLOT_DIR / "correlation_heatmap.png", bbox_inches="tight")
    plt.show()

    # 6. Variance plot (fixed)
    plt.figure(figsize=(12, 6))
    plt.plot(band_numbers, df[reflectance_cols].var())
    plt.title("Variance of Reflectance Across Wavelengths")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Variance")
    plt.savefig(PLOT_DIR / "variance_plot.png", bbox_inches="tight")
    plt.show()

    # 7. Smoothed spectral signatures (fixed)
    plt.figure(figsize=(12, 6))
    sample_indices = np.random.choice(df.index, size=5, replace=False)
    for idx in sample_indices:
        smoothed = savgol_filter(df.loc[idx, reflectance_cols], window_length=11, polyorder=2)
        plt.plot(band_numbers, smoothed, label=f"Sample {idx}")
    plt.title("Smoothed Spectral Signatures (Savitzky-Golay Filter)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Smoothed Reflectance")
    plt.legend()
    plt.savefig(PLOT_DIR / "smoothed_signatures.png", bbox_inches="tight")
    plt.show()

    # 8. Full heatmap (fixed)
    plt.figure(figsize=(14, 8))
    heatmap_data = df[reflectance_cols].copy()
    heatmap_data.columns = band_numbers
    sns.heatmap(heatmap_data.T, cmap="viridis", cbar_kws={'label': 'Reflectance'})
    plt.title("Spectral Band Heatmap Across Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Wavelength (nm)")
    plt.savefig(PLOT_DIR / "spectral_heatmap.png", bbox_inches="tight")
    plt.show()

    # 9. Pairplot (fixed)
    subset_bands = selected_bands[:4]
    pairplot_data = df[subset_bands].copy()
    pairplot_data.columns = [f"{float(col):.1f} nm" for col in pairplot_data.columns]
    sns.pairplot(pairplot_data)
    plt.suptitle("Pairplot of Selected Bands", y=1.02)
    plt.savefig(PLOT_DIR / "pairplot_selected_bands.png", bbox_inches="tight")
    plt.show()

def sensor_drift_check(df: pd.DataFrame, reflectance_cols: list) -> None:
    """
    Perform basic sensor drift check using rolling mean of reflectance values over sample index.
    Assumes 'hsi_id' column represents acquisition order.

    :param df: Raw DataFrame.
    :param reflectance_cols: List of spectral band columns.
    """
    try:
        logger.info("🔎 Performing sensor drift check...")

        # Use selected bands across the spectrum (beginning, middle, end)
        selected_bands = [reflectance_cols[0], reflectance_cols[len(reflectance_cols)//2], reflectance_cols[-1]]
        band_labels = ["Low", "Mid", "High"]

        plt.figure(figsize=(14, 6))
        for band, label in zip(selected_bands, band_labels):
            rolling_mean = df[band].rolling(window=20, center=True).mean()
            plt.plot(df["hsi_id"], rolling_mean, label=f"{label} Band ({band})")

        plt.title("Sensor Drift Check: Rolling Mean of Reflectance (Window=20)")
        plt.xlabel("HSI Sample ID (proxy for time)")
        plt.ylabel("Reflectance (Rolling Avg)")
        plt.legend()
        plt.grid(True)
        drift_plot_path = PLOT_DIR / "sensor_drift_check.png"
        plt.savefig(drift_plot_path, bbox_inches="tight")
        plt.close()

        logger.info(f"✅ Sensor drift visualization saved to: {drift_plot_path}")

    except Exception as e:
        logger.error(f"❗ Error during sensor drift check: {str(e)}")


def preprocess_data(df: pd.DataFrame,
                    reflectance_cols: list,
                    nir_band: str = "100",
                    red_band: str = "50") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Preprocess data, returning separate X_train, X_test, y_train, y_test.

    :param df: Raw DataFrame.
    :param reflectance_cols: List of reflectance columns.
    :param nir_band: Band name for NIR.
    :param red_band: Band name for Red.
    :return: (X_train_final, X_test_final, y_train_final, y_test_final)
    """
    # 1. Train-test split
    X = df.drop(columns=["vomitoxin_ppb"])
    y = df["vomitoxin_ppb"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    logger.info(f"✅ Split data: X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}")

    # 2. Add NDVI-like index post-split
    X_train["ndvi_like"] = (X_train[nir_band] - X_train[red_band]) / (X_train[nir_band] + X_train[red_band] + 1e-9)
    X_test["ndvi_like"] = (X_test[nir_band] - X_test[red_band]) / (X_test[nir_band] + X_test[red_band] + 1e-9)
    logger.info("✅ NDVI-like index added post-split.")

    # 3. Imputation and Scaling
    feature_cols = reflectance_cols + ["ndvi_like"]
    imputer = SimpleImputer(strategy="mean")
    X_train[feature_cols] = imputer.fit_transform(X_train[feature_cols])
    X_test[feature_cols] = imputer.transform(X_test[feature_cols])
    logger.info("✅ Missing values imputed with mean.")

    scaler = StandardScaler()
    X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test[feature_cols] = scaler.transform(X_test[feature_cols])
    logger.info("✅ Reflectance bands standardized.")

    # 4. Anomaly detection via Z-score
    z_scores = np.abs(zscore(X_train[reflectance_cols]))
    anomaly_mask = (z_scores > 3).sum(axis=1) < 0.5 * len(reflectance_cols)

    X_train = X_train[anomaly_mask].reset_index(drop=True)
    y_train = y_train[anomaly_mask].reset_index(drop=True)

    # 5. PCA for Dimensionality Reduction
    pca = PCA(n_components=70)
    X_train_processed = pca.fit_transform(X_train[feature_cols])
    X_test_processed = pca.transform(X_test[feature_cols])
    explained_var = pca.explained_variance_ratio_.sum()
    logger.info(f"✅ PCA applied: 70 components retained, explaining {explained_var:.2%} variance.")

    # PCA visualization
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.axhline(y=0.98, color='r', linestyle='--', label='98% Variance')
    plt.title("Cumulative Explained Variance by PCA Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_DIR / "pca_explained_variance.png", bbox_inches="tight")
    plt.show()

    # 6. Prepare final outputs
    X_train_final = pd.DataFrame(X_train_processed, columns=[f"PC{i+1}" for i in range(70)])
    X_train_final["hsi_id"] = X_train["hsi_id"].reset_index(drop=True)

    X_test_final = pd.DataFrame(X_test_processed, columns=[f"PC{i+1}" for i in range(70)])
    X_test_final["hsi_id"] = X_test["hsi_id"].reset_index(drop=True)

    y_train_final = pd.DataFrame({
        "hsi_id": X_train["hsi_id"].reset_index(drop=True),
        "vomitoxin_ppb": y_train
    })
    y_test_final = pd.DataFrame({
        "hsi_id": X_test["hsi_id"].reset_index(drop=True),
        "vomitoxin_ppb": y_test.reset_index(drop=True)
    })

    logger.info(f"Final shapes before return: X_train {X_train_final.shape}, "
                f"X_test {X_test_final.shape}, y_train {y_train_final.shape}, y_test {y_test_final.shape}")

    return X_train_final, X_test_final, y_train_final, y_test_final