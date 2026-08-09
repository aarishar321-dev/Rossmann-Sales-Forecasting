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

@st.cache_resource
def load_model():
    return joblib.load("models/forecast_rf_model_compressed.pkl")

with st.spinner("Loading preprocessor..."):
    preprocessor = load_preprocessor()

st.success("Preprocessor loaded successfully.")

with st.spinner("Loading forecasting model..."):
    model = load_model()

st.success("Random Forest model loaded successfully.")

st.write("Model type:", type(model))