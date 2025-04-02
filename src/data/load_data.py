"""
load_data.py

Responsible for loading the raw dataset from CSV,
and performing initial checks (negative reflectance, etc.).
"""

import pandas as pd
import logging

# Global variable to store reflectance column names
reflectance_cols = None
logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw data and check for inconsistencies.
    Updates the global variable `reflectance_cols`.
    Returns the loaded dataframe.
    """
    global reflectance_cols  # Use global to set reflectance_cols

    # Load CSV
    df = pd.read_csv(path)

    # Extract reflectance columns
    reflectance_cols = [
        col for col in df.columns if col not in [
            "hsi_id", "vomitoxin_ppb"]]
    logger.info(f"Reflectance columns detected: {len(reflectance_cols)} bands")

    # Check for negative values in reflectance columns
    negative_values = (df[reflectance_cols] < 0).sum().sum()
    if negative_values > 0:
        logger.warning(
            f"Warning: {negative_values} negative reflectance values found.")

    logger.info(f"Loaded data with shape: {df.shape}")
    logger.info(str(df.info()))

    # Return only the dataframe
    return df
