import numpy as np
import pandas as pd
import logging
from lightgbm import LGBMClassifier
from src.data_loader import load_data, preprocess_data
from src.utils import save_model
from typing import Optional, Union


def train_model(
    X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]
) -> LGBMClassifier:
    """
    Train a LightGBM model on the given dataset.

    Args:
        X (pd.DataFrame | np.ndarray): Feature matrix.
        y (pd.Series | np.ndarray): Target vector.

    Returns:
        LGBMClassifier: Trained LightGBM model.
    """
    try:
        logging.info("Initializing the LightGBM model...")
        model = LGBMClassifier(random_state=42)

        logging.info("Training the model...")
        model.fit(X, y)
        logging.info("Model training complete.")
        return model

    except Exception as e:
        logging.error(f"Error while training model: {e}", exc_info=True)
        raise


def training_pipeline(df: Optional[pd.DataFrame] = None) -> None:
    """
    Complete training pipeline: loads data, preprocesses it, trains the model, and saves it.

    Args:
        df (Optional[pd.DataFrame]): If provided, uses the given DataFrame. Otherwise, loads data.
    """
    try:
        if df is None:
            logging.info("No DataFrame provided, loading data...")
            df = load_data("data/soil_measures.csv")

        logging.info("Preprocessing data...")
        X, y = preprocess_data(df)

        logging.info("Starting model training...")
        model = train_model(X, y)

        logging.info("Saving the trained model...")
        save_model(model=model)
        logging.info("Training pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Training pipeline failed: {e}", exc_info=True)
        raise
