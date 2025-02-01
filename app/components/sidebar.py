import streamlit as st


def menubar():
    """Displays sidebar navigation menu."""
    st.sidebar.image("assets/logo.png", width=150)
    st.sidebar.page_link("Home.py", label="Home", icon="🏠")
    st.sidebar.page_link("pages/Analyze.py", label="Analyze", icon="📊")
    st.sidebar.page_link("pages/Train.py", label="Re-Train", icon="🛠️")
