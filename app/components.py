import streamlit as st

def sidebar():
    """Create a sidebar menu."""
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Train Model", "Predict"])
    return page
