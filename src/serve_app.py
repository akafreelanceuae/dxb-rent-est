import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import plotly.express as px

def create_price_distribution_plot():
    df = pd.read_csv("data/raw_listings.csv")
    fig = px.histogram(df[df['rent_aed'] <= 200000]['rent_aed'], 
                      nbins=50, title='Dubai Rent Price Distribution')
    fig.update_layout(xaxis_title='Rent (AED)', yaxis_title='Count')
    return fig

st.title("Dubai Rent Price Estimator")
st.sidebar.image(Image.open('dubai_skyline.jpg'), width=300)

# Show price distribution
st.plotly_chart(create_price_distribution_plot())

with st.form("rent_form"):
    col1, col2 = st.columns(2)
    
    # Property characteristics
    bedrooms = col1.number_input("Bedrooms", 0, 10, 2)
    bathrooms = col1.number_input("Bathrooms", 0, 10, 2)
    size_sqft = col1.number_input("Size (sq.ft)", 150, 20000, 1100)
    
    # Location features
    distance_to_metro_km = col2.slider("Distance to Metro (km)", 0.0, 20.0, 1.0)
    distance_to_mall_km = col2.slider("Distance to Mall (km)", 0.0, 20.0, 2.0)
    
    # Amenities
    amenities = []
    if st.checkbox("Pool"): amenities.append("pool")
    if st.checkbox("Gym"): amenities.append("gym")
    if st.checkbox("Balcony"): amenities.append("balcony")
    
    submitted = st.form_submit_button("Get Price Estimate")

if submitted:
    sample = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "size_sqft": size_sqft,
        "distance_to_metro_km": distance_to_metro_km,
        "distance_to_mall_km": distance_to_mall_km,
        "listing_month": 9,
        "listing_quarter": 3
    }])
    
    preprocessor = joblib.load("models/preprocessor.joblib")
    model = joblib.load("models/rent_model.joblib")
    
    X_transformed = preprocessor.transform(sample)
    prediction = model.predict(X_transformed)[0]
    
    st.metric("Predicted Monthly Rent", f"{prediction:,.0f} AED")