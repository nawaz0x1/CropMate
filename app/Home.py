import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import streamlit as st
from src.utils import load_model, predict_crop
from components.sidebar import menubar


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    filename="log.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
)

st.set_page_config(page_title="CropMateAI", page_icon="🌿")

menubar()
# Page title
st.title("🌾 AI Crop Recommendation")

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


# Load the trained model
try:
    model = load_model()
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    st.error("Failed to load the model. Please try again later.")

# User input for crop recommendation
nitrogen = st.number_input(
    "Nitrogen (N) Content (mg/kg)", min_value=0.0, max_value=150.0, value=70.0, step=1.0
)
phosphorus = st.number_input(
    "Phosphorus (P) Content (mg/kg)",
    min_value=0.0,
    max_value=150.0,
    value=50.0,
    step=1.0,
)
potassium = st.number_input(
    "Potassium (K) Content (mg/kg)",
    min_value=0.0,
    max_value=150.0,
    value=20.0,
    step=1.0,
)
soil_ph = st.number_input(
    "Soil Acidity (pH Level)", min_value=0.0, max_value=14.0, value=6.5, step=0.1
)

# Handle recommendation button click
if st.button("Recommend Crop"):
    try:
        logging.info("Recommendation button clicked.")
        crop, probability = predict_crop(
            model=model, n=nitrogen, p=phosphorus, k=potassium, ph=soil_ph
        )

        # Display the results
        if crop in emoji_dict:
            st.subheader(f"Recommended Crop: {crop.title()} {emoji_dict[crop]}")
        else:
            st.subheader(f"Recommended Crop: {crop.title()} 🫘")

        st.write(f"Probability: {probability:.2f}")

    except Exception as e:
        logging.error(f"Error during recommendation: {str(e)}")
        st.error(f"An error occurred during crop recommendation: {str(e)}")
