import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ============================================
# HOUSE PRICE PREDICTION APP
# ============================================

MODEL_PATH = "house_price_random_forest.pkl"

# Load trained model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find '{MODEL_PATH}'. "
        "Place the saved model file in the same folder as app.py."
    )
    st.stop()

# Exact feature order used during training
FEATURES = [
    "number of bedrooms",
    "number of bathrooms",
    "living area",
    "lot area",
    "number of floors",
    "waterfront present",
    "number of views",
    "condition of the house",
    "grade of the house",
    "Area of the house(excluding basement)",
    "Area of the basement",
    "Built Year",
    "Renovation Year",
    "Postal Code",
    "Lattitude",
    "Longitude",
    "living_area_renov",
    "lot_area_renov",
    "Number of schools nearby",
    "Distance from the airport",
    "Sale_Year",
    "Sale_Month"
]

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Header
st.title("🏠 House Price Predictor")
st.write(
    "Enter the property details below to estimate its market price "
    "using a trained Random Forest Regression model."
)

st.divider()

# ============================================
# BASIC PROPERTY INFORMATION
# ============================================

st.subheader("🏠 Basic Property Information")

col1, col2, col3 = st.columns(3)

with col1:
    bedrooms = st.number_input(
        "Number of bedrooms",
        min_value=0,
        value=3,
        step=1
    )

with col2:
    bathrooms = st.number_input(
        "Number of bathrooms",
        min_value=0.0,
        value=2.5,
        step=0.25
    )

with col3:
    floors = st.number_input(
        "Number of floors",
        min_value=0.0,
        value=2.0,
        step=0.5
    )

# ============================================
# AREA & SIZE
# ============================================

st.subheader("📐 Area & Size")

col1, col2 = st.columns(2)

with col1:
    living_area = st.number_input(
        "Living area",
        min_value=0,
        value=1800,
        step=100,
        help="Living area in the same unit used by the training dataset."
    )

with col2:
    lot_area = st.number_input(
        "Lot area",
        min_value=0,
        value=5000,
        step=100
    )

col1, col2 = st.columns(2)

with col1:
    house_area_excl_basement = st.number_input(
        "House area (excluding basement)",
        min_value=0,
        value=1800,
        step=100
    )

with col2:
    basement_area = st.number_input(
        "Basement area",
        min_value=0,
        value=0,
        step=100
    )

col1, col2 = st.columns(2)

with col1:
    living_area_renov = st.number_input(
        "Renovated living area",
        min_value=0,
        value=1800,
        step=100
    )

with col2:
    lot_area_renov = st.number_input(
        "Renovated lot area",
        min_value=0,
        value=5000,
        step=100
    )

# ============================================
# CONSTRUCTION & QUALITY
# ============================================

st.subheader("🏗️ Construction & Quality")

col1, col2, col3 = st.columns(3)

with col1:
    condition = st.number_input(
        "Condition of the house",
        min_value=1,
        max_value=5,
        value=3,
        step=1
    )

with col2:
    grade = st.number_input(
        "Grade of the house",
        min_value=1,
        max_value=13,
        value=8,
        step=1
    )

with col3:
    built_year = st.number_input(
        "Built Year",
        min_value=1800,
        max_value=2100,
        value=2000,
        step=1
    )

renovation_year = st.number_input(
    "Renovation Year (enter 0 if never renovated)",
    min_value=0,
    max_value=2100,
    value=0,
    step=1
)

# ============================================
# LOCATION
# ============================================

st.subheader("📍 Location")

col1, col2, col3 = st.columns(3)

with col1:
    postal_code = st.number_input(
        "Postal Code",
        min_value=0,
        value=98001,
        step=1
    )

with col2:
    latitude = st.number_input(
        "Latitude",
        value=47.35,
        format="%.6f"
    )

with col3:
    longitude = st.number_input(
        "Longitude",
        value=-122.20,
        format="%.6f"
    )

# ============================================
# AMENITIES & SURROUNDINGS
# ============================================

st.subheader("🌳 Amenities & Surroundings")

col1, col2, col3 = st.columns(3)

with col1:
    waterfront = st.selectbox(
        "Waterfront present",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes"
    )

with col2:
    views = st.number_input(
        "Number of views",
        min_value=0,
        value=0,
        step=1
    )

with col3:
    schools_nearby = st.number_input(
        "Number of schools nearby",
        min_value=0,
        value=2,
        step=1
    )

distance_airport = st.number_input(
    "Distance from the airport",
    min_value=0.0,
    value=30.0,
    step=1.0
)

# ============================================
# SALE INFORMATION
# ============================================

st.subheader("📅 Sale Information")

sale_date = st.date_input(
    "Sale date",
    value=date(2014, 5, 1)
)

sale_year = sale_date.year
sale_month = sale_date.month

st.caption(
    f"Model inputs generated automatically: "
    f"Sale Year = {sale_year}, Sale Month = {sale_month}"
)

st.divider()

# ============================================
# PREDICTION
# ============================================

if st.button("🔮 Predict House Price", use_container_width=True):

    # Basic validation
    if renovation_year != 0 and renovation_year < built_year:
        st.warning(
            "The renovation year is earlier than the built year. "
            "Please check the entered values."
        )
        st.stop()

    # Create input DataFrame in EXACT training order
    input_data = pd.DataFrame([{
        "number of bedrooms": bedrooms,
        "number of bathrooms": bathrooms,
        "living area": living_area,
        "lot area": lot_area,
        "number of floors": floors,
        "waterfront present": waterfront,
        "number of views": views,
        "condition of the house": condition,
        "grade of the house": grade,
        "Area of the house(excluding basement)": house_area_excl_basement,
        "Area of the basement": basement_area,
        "Built Year": built_year,
        "Renovation Year": renovation_year,
        "Postal Code": postal_code,
        "Lattitude": latitude,
        "Longitude": longitude,
        "living_area_renov": living_area_renov,
        "lot_area_renov": lot_area_renov,
        "Number of schools nearby": schools_nearby,
        "Distance from the airport": distance_airport,
        "Sale_Year": sale_year,
        "Sale_Month": sale_month
    }], columns=FEATURES)

    # Predict
    prediction = model.predict(input_data)[0]

    # Display result
    st.success("Prediction generated successfully!")

    st.metric(
        label="Estimated House Price",
        value=f"${prediction:,.2f}"
    )

    st.info(
        "This estimate is produced by the trained Random Forest model "
        "and should be treated as a prediction, not a guaranteed market price."
    )

# ============================================
# MODEL INFORMATION
# ============================================

with st.expander("ℹ️ About the Model"):
    st.write(
        "This application uses a Random Forest Regression model trained "
        "on the House Price India dataset."
    )

    st.write("**Model configuration:**")
    st.code(
        """RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)""",
        language="python"
    )

    st.write("**Test-set performance:**")
    st.write("- R² Score: 0.8747 (87.47%)")
    st.write("- MAE: 68,636.02")
    st.write("- RMSE: 132,893.75")

    st.write(
        "**5-fold cross-validation:** mean R² = 0.9351 ± 0.0038"
    )
