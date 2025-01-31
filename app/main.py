import sys
import os

# Add the root directory (ML-App) to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import pandas as pd
from src.data_loader import load_data, preprocess_data
from src.model import train_model
from src.utils import load_model

st.title("ML App 🚀")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = load_data(uploaded_file)
    df = preprocess_data(df)
    st.write("Preview of Data:", df.head())

    if st.button("Train Model"):
        model = train_model(df)
        st.success("Model trained successfully!")

if st.button("Load & Predict"):
    model = load_model()
    st.write("Model loaded. Ready for predictions!")
