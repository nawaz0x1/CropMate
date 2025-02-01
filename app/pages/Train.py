import streamlit as st
import pandas as pd
import logging
from src.model import training_pipeline
from components.sidebar import menubar

menubar()
st.title("Train on New Data")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        logging.info("File uploaded successfully.")

        df = pd.read_csv(uploaded_file)

        st.write("Preview of Data:", df.head())
        logging.info("Data preview displayed.")

        progress_bar = st.progress(0)

        if st.button("Train Model"):
            try:
                progress_bar.progress(20)
                logging.info("Training started.")

                training_pipeline(df)

                progress_bar.progress(100)
                logging.info("Model training completed successfully.")

                st.success("Model trained successfully!")

                try:
                    st.cache_resource.clear()
                    st.cache_data.clear()
                    logging.info("Cache cleared successfully!")
                except Exception as e:
                    logging.error(f"Error during cache clearance: {str(e)}")

            except Exception as e:
                logging.error(f"Error during training: {str(e)}")
                st.error(
                    "Something went wrong during model training. Please try again."
                )
                raise e

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        st.error(
            "An error occurred while processing the file. Please check the format and try again."
        )
