import streamlit as st
import pandas as pd
import logging
from src.data_loader import load_data, preprocess_data
from src.model import train_model, training_pipeline
from src.utils import load_model

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.title("Train on New Data")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        logger.info("File uploaded successfully.")
        
        df = pd.read_csv(uploaded_file)
        
        st.write("Preview of Data:", df.head())
        logger.info("Data preview displayed.")
        
        progress_bar = st.progress(0)
        
        if st.button("Train Model"):
            try:
                progress_bar.progress(20)
                logger.info("Training started.")
                
                training_pipeline(df)
                
                progress_bar.progress(100)
                logger.info("Model training completed successfully.")
                
                st.success("Model trained successfully!")
            
            except Exception as e:
                logger.error(f"Error during training: {str(e)}")
                st.error("Something went wrong during model training. Please try again.")
                raise e 

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error("An error occurred while processing the file. Please check the format and try again.")
