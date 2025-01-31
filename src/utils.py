import os
import joblib
import logging
from typing import Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_PATH = "models/model.pkl"

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
