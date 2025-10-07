import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import datetime
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any

# --- Page Configuration ---
st.set_page_config(page_title="Commercial Rent Predictor", layout="wide", initial_sidebar_state="expanded")

# --- Constants and Options (from your original script) ---
PROPERTY_TYPES = ['showroom', 'shop', 'bare shell office', 'ready to use office', 
                 'commercial property', 'werehouse', 'godown']
AREAS = ['manewada', 'jaitala', 'besa', 'omkar nagar', 'itwari', 'hingna', 
        'sitabuldi', 'mahal', 'kharbi', 'mihan', 'pratap nagar', 'ramdaspeth', 
        'dharampeth', 'gandhibag', 'chatrapati nagar', 'nandanwan', 'sadar', 
        'dighori', 'somalwada', 'ganeshpeth colony', 'mhalgi nagar', 'sakkardara', 
        'babulban', 'manish nagar', 'dhantoli', 'khamla', 'laxminagar', 'ajni', 
        'wathoda', 'hulkeshwar', 'pardi', 'new indora', 'civil lines', 'gadhibag', 
        'bagadganj', 'swawlambi nagar', 'manawada', 'trimurti nagar', 'lakadganj', 'shivaji nagar']
ZONES = ['south', 'west', 'east', 'north']
LOCATION_HUBS = ['commercial project', 'others', 'retail complex/building', 
                'market/high street', 'business park', 'it park', 'residential']
OWNERSHIPS = ['freehold', 'leasehold', 'cooperative society', 'power_of_attorney']
POSSESSION_STATUSES = ['ready to move', 'Under Construction']
POSTED_BY_OPTIONS = ['owner', 'housing expert', 'broker']
YES_NO_OPTIONS = ['yes', 'no']
FLOOR_OPTIONS = ['ground floor', '1 floor', '2 floor', '1, 2,3 floors', 
                'ground floor,1 floor', '1,2,3 floors', '1,2 floors', 
                '1,2,3,4,GF', '1 , GF floor', '8 floor', '3 floor']
TOTAL_FLOORS_OPTIONS = ['3 floors', '1 floor', '2 floors', '4 floors', 
                       '5 floors', '8 floors', '7 floors', '6 floors', 
                       '15 floors', '9 floors', '10 floors']
AMENITIES_OPTIONS = ['parking', 'vastu', 'lift', 'cabin', 'meeting room', 'dg and ups', 
                    'water storage', 'staircase', 'security', 'cctv', 'power backup', 
                    'reception area', 'pantry', 'fire extinguishers', 'fire safety', 
                    'oxygen duct', 'food court', 'furnishing', 'internet', 'fire sensors']
LOCK_IN_PERIOD_OPTIONS = ['2 months', '6 months', '12 months', '3 months', '1 month', 
                         '11 months', '4 months', '10 months', '6  months', '8  months', 
                         '4  months', '36 months']
EXPECTED_RENT_INCREASE_OPTIONS = ['0.05', '0.10']

# --- Model Loading Function ---
@st.cache_resource(show_spinner="Loading model and resources...")
def load_model_components():
    """Loads the trained model, scaler, and feature names."""
    try:
        model = joblib.load('mc.pkl')
        scaler = joblib.load('sc.pkl')
        feature_names = joblib.load('fc.pkl')
        return model, scaler, list(feature_names)
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'mc.pkl', 'sc.pkl', and 'fc.pkl' are in the same directory as the app.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading model components: {e}")
        return None, None, None

# --- Preprocessing Function (identical to your script) ---
def preprocess_input(df: pd.DataFrame, feature_names: List[str], scaler: Any) -> pd.DataFrame:
    """Preprocesses user input to match the model's training data format."""
    processed_df = df.copy()
    
    # Process floor_no
    def floor_to_int_list(floor_str):
        floor_str = str(floor_str).lower().replace('floor', '').replace('floors', '').replace(' ', '')
        floor_str = floor_str.replace('ground', '0').replace('gf', '0')
        parts = re.split(r'[,]', floor_str)
        floor_numbers = []
        for p in parts:
            try:
                floor_numbers.append(int(p))
            except ValueError:
                continue
        return ','.join(map(str, sorted(floor_numbers))) if floor_numbers else None
    
    processed_df['floor_no'] = processed_df['floor_no'].apply(floor_to_int_list)
    
    # Process total_floors
    def total_floors_to_int(floor_str):
        try:
            return int(str(floor_str).lower().replace('floors', '').replace('floor', '').strip())
        except ValueError:
            return None
    
    processed_df['total_floors'] = processed_df['total_floors'].apply(total_floors_to_int)
    
    # Process size columns
    def size_to_int(size_str):
        try:
            return int(str(size_str).lower().replace('sqft','').replace('sq.ft','').strip())
        except ValueError:
            return None
    
    processed_df['size_in_sqft'] = processed_df['size_in_sqft'].apply(size_to_int)
    processed_df['carpet_area_sqft'] = processed_df['carpet_area_sqft'].apply(size_to_int)
    
    # Process amenities into binary columns
    def extract_amenities_list(text):
        text = str(text).lower()
        text = re.sub(r'\(\d+\)', '', text)
        amenities = [x.strip() for x in text.split(',') if x.strip()]
        return amenities

    all_amenities = AMENITIES_OPTIONS
    for amenity in all_amenities:
        amenity_col = amenity.replace(' ', '_')
        processed_df[amenity_col] = processed_df['amenities_count'].apply(
            lambda x: 1 if amenity in extract_amenities_list(x) else 0
        )
    
    # Process property_age and lock_in_period
    processed_df['property_age'] = processed_df['property_age'].astype(int)
    processed_df['lock_in_period_in_months'] = processed_df['lock in period'].str.replace('months', '', regex=False) \
                                           .str.replace('month', '', regex=False) \
                                           .str.strip().fillna(0).astype(int)
    
    # One-hot encode categorical features
    categorical_features = ['listing litle', 'city', 'area', 'zone', 'location_hub',
                           'property_type', 'ownership', 'floor_no',
                           'electric_charge_included', 'water_charge_included',
                           'possession_status', 'posted_by', 'negotiable', 'brokerage']
    
    for feature in categorical_features:
        if feature in processed_df.columns:
            # CRITICAL FIX: Removed 'prefix=feature' to exactly match the original training script's column names.
            # This ensures a column like '0' (for ground floor) is created, not 'floor_no_0'.
            processed_df = pd.get_dummies(processed_df, columns=[feature])

    # Align columns with training data
    for col in feature_names:
        if col not in processed_df.columns:
            processed_df[col] = 0
    
    # Ensure correct column order
    processed_df = processed_df[feature_names]
    
    # Scale numerical features
    numerical_cols_for_scaling = [
        'size_in_sqft', 'carpet_area_sqft', 'private_washroom', 'public_washroom',
        'total_floors', 'property_age', 'expected rent increases yearly',
        'lock_in_period_in_months'
    ]
    numerical_cols_for_scaling.extend([amenity.replace(' ', '_') for amenity in all_amenities])
    numerical_cols_present = [col for col in numerical_cols_for_scaling if col in processed_df.columns]
    
    if numerical_cols_present:
        processed_df[numerical_cols_present] = scaler.transform(processed_df[numerical_cols_present])
    
    return processed_df

# --- Main App Logic ---
def main():
    model, scaler, features = load_model_components()
    
    st.title("🏢 Commercial Property Rent Prediction")
    st.markdown("Enter the details of the commercial property to get an estimated rent price.")

    if model is None:
        st.stop() # Stop execution if model loading failed

    # --- Input Form ---
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("Basic & Location Details")
            property_type = st.selectbox("Property Type*", PROPERTY_TYPES)
            size_sqft = st.number_input("Size in sqft*", min_value=100, value=2000, step=50)
            carpet_area = st.number_input("Carpet Area in sqft*", min_value=100, value=1800, step=50)
            
            area = st.selectbox("Area*", AREAS)
            zone = st.selectbox("Zone*", ZONES)
            location_hub = st.selectbox("Location Hub*", LOCATION_HUBS)
            ownership = st.selectbox("Ownership Type*", OWNERSHIPS)
            
            st.header("Features & Other Details")
            private_washroom = st.number_input("Number of Private Washrooms", min_value=0, value=1, step=1)
            public_washroom = st.number_input("Number of Public Washrooms", min_value=0, value=1, step=1)
            floor_no = st.selectbox("Floor Number*", FLOOR_OPTIONS)
            total_floors = st.selectbox("Total Floors in Building*", TOTAL_FLOORS_OPTIONS)
            property_age = st.number_input("Property Age (years)*", min_value=0, value=5, step=1)

        with col2:
            st.header("Amenities & Charges")
            amenities = st.multiselect("Select Amenities", AMENITIES_OPTIONS)
            electric_charge = st.selectbox("Electric Charge Included*", YES_NO_OPTIONS)
            water_charge = st.selectbox("Water Charge Included*", YES_NO_OPTIONS)
            
            possession_status = st.selectbox("Possession Status*", POSSESSION_STATUSES)
            posted_by = st.selectbox("Posted By*", POSTED_BY_OPTIONS)
            lock_in_period_str = st.selectbox("Lock-in Period*", LOCK_IN_PERIOD_OPTIONS)
            expected_rent_increase_str = st.selectbox("Expected Yearly Rent Increase*", EXPECTED_RENT_INCREASE_OPTIONS)
            negotiable = st.selectbox("Negotiable*", YES_NO_OPTIONS)
            brokerage = st.selectbox("Brokerage*", YES_NO_OPTIONS)

        # --- Projection and Comparison Inputs ---
        st.markdown("---")
        st.header("Analysis & Projection")
        col3, col4, col5 = st.columns(3)
        with col3:
            listed_price = st.number_input("Listed Price (for comparison)", min_value=0, value=100000, step=5000)
        with col4:
            projection_years = st.slider("Project rent for (years)", min_value=1, max_value=20, value=5)
        with col5:
            annual_growth_rate = st.slider("Annual Growth Rate (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.5)
        
        # --- Submit Button ---
        submitted = st.form_submit_button("Predict Rent Price", use_container_width=True)

    # --- Prediction and Results ---
    if submitted:
        # Create a dictionary with user input, matching the original script's keys
        user_data = {
            'listing litle': property_type, # Note: listing litle is same as property_type
            'city': 'nagpur',
            'area': area,
            'zone': zone,
            'location_hub': location_hub,
            'property_type': property_type,
            'ownership': ownership,
            'size_in_sqft': size_sqft,
            'carpet_area_sqft': carpet_area,
            'private_washroom': private_washroom,
            'public_washroom': public_washroom,
            'floor_no': floor_no,
            'total_floors': total_floors,
            'amenities_count': ', '.join(amenities),
            'electric_charge_included': electric_charge,
            'water_charge_included': water_charge,
            'property_age': property_age,
            'possession_status': possession_status,
            'posted_by': posted_by,
            'lock in period': lock_in_period_str,
            'expected rent increases yearly': float(expected_rent_increase_str),
            'negotiable': negotiable,
            'brokerage': brokerage
        }
        
        # Convert to DataFrame
        user_df = pd.DataFrame([user_data])
        
        # Preprocess the data
        processed_df = preprocess_input(user_df, features, scaler)
        
        # Make prediction
        try:
            prediction_log = model.predict(processed_df)[0]
            prediction = np.expm1(prediction_log) # Convert back from log scale
            
            st.markdown("---")
            st.header("📊 Prediction Results")
            st.success(f"**Estimated Rent Price: ₹{prediction:,.2f}**")

            # Price Comparison
            st.subheader("Price Comparison")
            fair_price_tolerance = 0.25 # 25%
            lower_bound = prediction * (1 - fair_price_tolerance)
            upper_bound = prediction * (1 + fair_price_tolerance)
            
            st.write(f"**Fair Price Range (±{fair_price_tolerance*100:.0f}%):** ₹{lower_bound:,.2f} - ₹{upper_bound:,.2f}")
            st.write(f"**Your Listed Price:** ₹{listed_price:,.2f}")
            
            if listed_price < lower_bound:
                st.warning("🔻 The listed price seems to be **UNDERPRICED**.")
            elif listed_price > upper_bound:
                st.warning("🔺 The listed price seems to be **OVERPRICED**.")
            else:
                st.success("✅ The listed price appears to be **FAIR**.")

            # Future Projection
            st.subheader(f"Future Rent Projection ({projection_years} years)")
            future_rent = prediction * ((1 + annual_growth_rate / 100) ** projection_years)
            st.info(f"Projected rent in {projection_years} years: **₹{future_rent:,.2f}**")
            
            # 15-Year Projection Plot
            st.subheader("15-Year Projection Trend")
            years = np.arange(1, 16)
            projected_rents = [prediction * ((1 + annual_growth_rate / 100) ** y) for y in years]
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(years, projected_rents, marker='o', linestyle='-', color='#1f77b4')
            ax.set_title(f'15-Year Rent Projection at {annual_growth_rate}% Annual Growth')
            ax.set_xlabel("Year")
            ax.set_ylabel("Projected Rent (₹)")
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            ax.set_xticks(years)
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
