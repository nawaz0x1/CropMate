import os
import logging
import pandas as pd
import streamlit as st
from typing import Tuple


DEFAULT_FEATURES = ["N", "P", "K", "ph"]
DEFAULT_TARGET = "crop"


@st.cache_resource
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        if not os.path.exists(filepath):
            logging.error(f"File not found: {filepath}")
            raise FileNotFoundError(f"File not found: {filepath}")

        logging.info(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        logging.info("Data loaded successfully.")
        return df

    except Exception as e:
        logging.error(f"Error loading data: {e}", exc_info=True)
        raise


def preprocess_data(
    df: pd.DataFrame, features: list = DEFAULT_FEATURES, target: str = DEFAULT_TARGET
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the dataset: remove NaNs, extract features and target.

    Args:
        df (pd.DataFrame): The input dataset.
        features (list): List of feature column names.
        target (str): The target column name.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Processed feature matrix (X) and target vector (y).
    """
    try:
        logging.info("Preprocessing data...")

        # Ensure required columns exist
        missing_cols = [col for col in features + [target] if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Drop missing values
        df = df.dropna()

        X = df[features]
        y = df[target]

        logging.info("Data preprocessing completed successfully.")
        return X, y

    except Exception as e:
        logging.error(f"Error preprocessing data: {e}", exc_info=True)
        raise
