import joblib

def load_model():
    """Load a trained model."""
    return joblib.load("models/model.pkl")
