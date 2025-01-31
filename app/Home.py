import sys
import os
import logging
import shap
import pandas as pd
import numpy as np
import streamlit as st
from src.utils import load_model

# Add the root directory (ML-App) to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Page title
st.title("🌾 Crop Recommendation System")

# Crop emoji dictionary for display
emoji_dict = {
    "apple": "🍎",
    "banana": "🍌",
    "chickpea": "🧆",
    "coconut": "🥥",
    "coffee": "🍵",
    "cotton": "☁️",
    "grapes": "🍇",
    "jute": "🧵",
    "kidneybeans": "🫘",
    "lentil": "🥘",
    "maize": "🌽",
    "mango": "🥭",
    "mungbean": "🫛",
    "muskmelon": "🍈",
    "orange": "🍊",
    "papaya": "🏉",
    "pigeonpeas": "🫛",
    "pomegranate": "🌰🔴",
    "rice": "🌾",
    "watermelon": "🍉",
}

def predict_crop(model, N, P, K, ph):
    """Function to predict the crop based on input soil nutrients."""
    try:
        input_data = pd.DataFrame([[N, P, K, ph]], columns=["N", "P", "K", "ph"])
        probabilities = model.predict_proba(input_data)
        
        crop_index = probabilities[0].argmax()
        crop = model.classes_[crop_index]
        probability = probabilities[0][crop_index]

        logger.info(f"Prediction successful. Crop: {crop}, Probability: {probability:.2f}")
        return crop, probability
    except Exception as e:
        logger.error(f"Error in predicting crop: {str(e)}")
        raise RuntimeError("Error during crop prediction.")

# Load the trained model
try:
    model = load_model()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    st.error("Failed to load the model. Please try again later.")

# User input for crop recommendation
N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=100.0, step=0.1)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=100.0, step=0.1)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=100.0, step=0.1)
ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, step=0.1)

# Handle recommendation button click
if st.button("Recommend Crop"):
    try:
        logger.info("Recommendation button clicked.")
        crop, probability = predict_crop(model, N, P, K, ph)

        # Display the results
        if crop in emoji_dict:
            st.subheader(f"Recommended Crop: {crop.title()} {emoji_dict[crop]}")
        else:
            st.subheader(f"Recommended Crop: {crop.title()} 🫘")

        st.write(f"Probability: {probability:.2f}")

    except Exception as e:
        logger.error(f"Error during recommendation: {str(e)}")
        st.error(f"An error occurred during crop recommendation: {str(e)}")
