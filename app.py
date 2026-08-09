import streamlit as st
import joblib

st.set_page_config(
    page_title="Rossmann Sales Forecast",
    layout="wide"
)

st.title("Rossmann Sales Forecast")
st.write("APP STARTED")

@st.cache_resource
def load_preprocessor():
    return joblib.load("models/forecast_preprocessor.pkl")

with st.spinner("Loading preprocessor..."):
    preprocessor = load_preprocessor()

st.success("Preprocessor loaded successfully.")
st.write(type(preprocessor))