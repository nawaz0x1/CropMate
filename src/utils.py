import os
import joblib
import logging
import numpy as np
import pandas as pd
import streamlit as st
from typing import Any, Optional
from sklearn.base import ClassifierMixin

MODEL_PATH = "models/model.pkl"


@st.cache_resource
def load_model(model_path: str = MODEL_PATH) -> Optional[Any]:
    """
    Load a trained model from a file.

    Args:
        model_path (str): Path to the saved model.

    Returns:
        The loaded model if successful, otherwise None.
    """
    try:
        if not os.path.exists(model_path):
            logging.error(f"Model file not found at {model_path}")
            return None

        logging.info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model

    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        return None


def save_model(model: Any, model_path: str = MODEL_PATH) -> bool:
    """
    Save the trained model to a file.

    Args:
        model (Any): Trained model to save.
        model_path (str): Path where the model will be saved.

    Returns:
        bool: True if the model was saved successfully, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logging.info(f"Saving model to {model_path}...")
        joblib.dump(model, model_path)
        logging.info("Model saved successfully.")
        return True

    except Exception as e:
        logging.error(f"Error saving model: {e}", exc_info=True)
        return False


def predict_crop(
    model: ClassifierMixin, n: float, p: float, k: float, ph: float
) -> tuple[str, float]:
    """Predict the crop based on soil nutrient levels using the given model.

    Args:
        model (ClassifierMixin): Trained classification model.
        N (float): Nitrogen content.
        P (float): Phosphorus content.
        K (float): Potassium content.
        ph (float): Soil pH level.

    Returns:
        tuple[str, float]: Predicted crop and its probability.
    """
    try:
        input_data = pd.DataFrame([[n, p, k, ph]], columns=["N", "P", "K", "ph"])

        # Ensure model supports probability predictions
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "The provided model does not support probability prediction."
            )

        probabilities = model.predict_proba(input_data)
        crop_index = int(np.argmax(probabilities[0]))
        crop = str(model.classes_[crop_index])
        probability = float(probabilities[0][crop_index])

        logging.info(
            f"Prediction successful: Input={input_data.to_dict(orient='records')[0]}, Crop={crop}, Probability={probability:.2f}"
        )
        return crop, probability

    except AttributeError as e:
        logging.error(f"Model error: {e}")
        raise ValueError("Invalid model. Ensure it supports `predict_proba`.")

    except Exception as e:
        logging.exception("Unexpected error during crop prediction.")
        raise RuntimeError("Error during crop prediction.") from e
