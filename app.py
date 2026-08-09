import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Rossmann Sales Forecast",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */
    .stApp {
        background: #F5F7FA;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .hero {
        background: linear-gradient(
            135deg,
            #0B1F33 0%,
            #123B5D 60%,
            #0E7490 100%
        );
        padding: 42px 48px;
        border-radius: 18px;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(11, 31, 51, 0.15);
    }

    .hero h1 {
        color: white;
        font-size: 42px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -1px;
    }

    .hero p {
        color: #C7DCE8;
        font-size: 17px;
        margin-top: 10px;
        margin-bottom: 0;
    }

    /* Section headings */
    .section-title {
        color: #0B1F33;
        font-size: 22px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    /* Prediction card */
    .prediction-card {
        background: white;
        border-radius: 18px;
        padding: 32px;
        text-align: center;
        border: 1px solid #DDE5EC;
        box-shadow: 0 8px 25px rgba(11, 31, 51, 0.08);
        margin-top: 25px;
    }

    .prediction-label {
        color: #64748B;
        font-size: 15px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-value {
        color: #0E7490;
        font-size: 42px;
        font-weight: 750;
        margin-top: 8px;
    }

    .prediction-note {
        color: #64748B;
        font-size: 14px;
        margin-top: 8px;
    }

    /* Input labels */
    label {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* Primary button */
    div.stButton > button {
        width: 100%;
        background: #0E7490;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 13px 20px;
        font-size: 16px;
        font-weight: 650;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: #155E75;
        border: none;
        color: white;
    }

    /* Info box */
    .info-box {
        background: #E8F4F8;
        border-left: 4px solid #0E7490;
        padding: 15px 18px;
        border-radius: 8px;
        color: #164E63;
        margin-top: 20px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748B;
        font-size: 13px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #DDE5EC;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "models/forecast_xgb_model.pkl"
    )

    preprocessor = joblib.load(
        "models/forecast_preprocessor.pkl"
    )

    return model, preprocessor


with st.spinner("Loading the forecasting model..."):
    model, preprocessor = load_model()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Rossmann Sales Forecast</h1>
        <p>
            Store-level sales forecasting powered by machine learning.
            Enter the store and trading conditions to estimate daily sales.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Forecast Inputs</div>',
    unsafe_allow_html=True
)

left, right = st.columns(2, gap="large")


with left:

    store = st.number_input(
        "Store ID",
        min_value=1,
        max_value=1115,
        value=1,
        step=1
    )

    forecast_date = st.date_input(
        "Forecast Date"
    )

    day_of_week = st.selectbox(
        "Day of Week",
        options=[1, 2, 3, 4, 5, 6, 7],
        index=0,
        help="1 = Monday, 7 = Sunday"
    )

    open_store = st.selectbox(
        "Store Status",
        options=[1, 0],
        format_func=lambda x:
            "Open" if x == 1 else "Closed"
    )


with right:

    promo = st.selectbox(
        "Promotion",
        options=[0, 1],
        format_func=lambda x:
            "Active" if x == 1 else "Not Active"
    )

    state_holiday = st.selectbox(
        "State Holiday",
        options=["0", "a", "b", "c"],
        format_func=lambda x: {
            "0": "No State Holiday",
            "a": "Public Holiday",
            "b": "Easter Holiday",
            "c": "Christmas Holiday"
        }[x]
    )

    school_holiday = st.selectbox(
        "School Holiday",
        options=[0, 1],
        format_func=lambda x:
            "Yes" if x == 1 else "No"
    )


# ============================================================
# DATE FEATURES
# ============================================================

forecast_date = pd.Timestamp(forecast_date)

year = forecast_date.year
month = forecast_date.month
day = forecast_date.day
week = int(forecast_date.isocalendar().week)
quarter = forecast_date.quarter
is_weekend = int(forecast_date.dayofweek >= 5)


# ============================================================
# INPUT DATA
# ============================================================

input_data = pd.DataFrame({
    "Store": [store],
    "DayOfWeek": [day_of_week],
    "Open": [open_store],
    "Promo": [promo],
    "StateHoliday": [state_holiday],
    "SchoolHoliday": [school_holiday],
    "Year": [year],
    "Month": [month],
    "Day": [day],
    "Week": [week],
    "Quarter": [quarter],
    "IsWeekend": [is_weekend]
})


# ============================================================
# PREDICTION SECTION
# ============================================================

st.markdown(
    '<div class="section-title">Sales Prediction</div>',
    unsafe_allow_html=True
)


# ============================================================
# SINGLE PREDICT BUTTON
# ============================================================

generate_forecast = st.button(
    "Predict"
)


# ============================================================
# GENERATE PREDICTION
# ============================================================

if generate_forecast:

    if open_store == 0:

        prediction = 0

    else:

        processed_data = preprocessor.transform(
            input_data
        )

        prediction = model.predict(
            processed_data
        )[0]

        prediction = max(0, prediction)


    # ========================================================
    # PREDICTION CARD
    # ========================================================

    st.markdown(
        f"""<div class="prediction-card">
<div class="prediction-label">
Estimated Daily Sales
</div>

<div class="prediction-value">
{prediction:,.2f}
</div>

<div class="prediction-note">
Store {store} &nbsp;·&nbsp; {forecast_date.strftime("%d %B %Y")}
</div>

</div>""",
        unsafe_allow_html=True
    )

# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown(
    """<div class="info-box">
<b>Model:</b> XGBoost Regressor
&nbsp;&nbsp;|&nbsp;&nbsp;
<b>Forecast Features:</b> 12
</div>""",
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Rossmann Sales Forecasting · Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)