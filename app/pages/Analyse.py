import pandas as pd
import streamlit as st
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_data, preprocess_data

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Crop emoji dictionary
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

st.title("🌾 Crop Data Analysis")


try:
    df = load_data("data/soil_measures.csv")
    logger.info("Dataset loaded successfully.")
    # Check if the required columns are present
    required_columns = ["N", "P", "K", "ph", "crop"]
    if not all(col in df.columns for col in required_columns):
        st.error(
            f"CSV must contain the following columns: {', '.join(required_columns)}"
        )

    else:
        crops = df["crop"].unique()
        selected_crop = st.selectbox("Select Crop", crops)

        crop_data = df[df["crop"] == selected_crop]
        # Show average N, P, K, ph for the selected crop
        avg_values = crop_data[["N", "P", "K", "ph"]].mean()
        st.subheader(
            f"Average Nutrient Values for {selected_crop.title()} {emoji_dict.get(selected_crop, '🫘')}:"
        )
        st.write(f"Nitrogen (N): {avg_values['N']:.2f}")
        st.write(f"Phosphorus (P): {avg_values['P']:.2f}")
        st.write(f"Potassium (K): {avg_values['K']:.2f}")
        st.write(f"pH Value: {avg_values['ph']:.2f}")

        # Plotting distribution for each nutrient
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        sns.histplot(crop_data["N"], kde=True, ax=axes[0, 0], color="green").set_title(
            "Nitrogen (N) Distribution"
        )
        sns.histplot(crop_data["P"], kde=True, ax=axes[0, 1], color="orange").set_title(
            "Phosphorus (P) Distribution"
        )
        sns.histplot(crop_data["K"], kde=True, ax=axes[1, 0], color="blue").set_title(
            "Potassium (K) Distribution"
        )
        sns.histplot(
            crop_data["ph"], kde=True, ax=axes[1, 1], color="purple"
        ).set_title("pH Value Distribution")

        plt.tight_layout()
        st.pyplot(fig)
except Exception as e:
    logger.error(f"Error loading or processing file: {str(e)}")
    st.error(f"An error occurred: {str(e)}")
