import streamlit as st
import joblib
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Commercial Property Rent Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the saved model and components
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load('mc.pkl')
        scaler = joblib.load('sc.pkl')
        feature_names = joblib.load('fc.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None

model, scaler, feature_names = load_model_components()

# Define helper functions
def floor_to_int_list(floor_str):
    floor_str = str(floor_str).lower()
    floor_str = floor_str.replace('floor', '').replace('floors', '').replace(' ', '')
    floor_str = floor_str.replace('ground', '0').replace('gf', '0')
    parts = re.split(r'[,]', floor_str)
    floor_numbers = []
    for p in parts:
        try:
            floor_numbers.append(int(p))
        except:
            continue
    if floor_numbers:
        return ','.join(map(str, sorted(floor_numbers)))
    return '0'

def process_user_input(inputs):
    # Create a DataFrame with all features initialized to 0
    input_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Set numerical values
    numerical_cols = ['size_in_sqft', 'carpet_area_sqft', 'private_washroom', 
                     'public_washroom', 'total_floors', 'property_age',
                     'expected rent increases yearly', 'lock_in_period_in_months']
    
    for col in numerical_cols:
        if col in inputs:
            input_df[col] = inputs[col]
    
    # Set amenities
    amenities_cols = ['fire_extinguishers', 'food_court', 'cabin', 'lift', '0', 
                     'water_storage', 'dg', 'fire_safety', 'security', 'cctv', 
                     'oxygen_duct', 'furnishing', 'vastu', 'reception_area', 
                     'internet', 'water_supply', 'fire_sensors', 'power_backup', 
                     'dg_and_ups', 'parking', 'pantry']
    
    for amenity in inputs.get('amenities', []):
        amenity_clean = amenity.lower().replace(' ', '_')
        if amenity_clean in amenities_cols:
            input_df[amenity_clean] = 1
    
    # Set categorical features
    categorical_mappings = {
        'listing litle': inputs['listing litle'],
        'area': inputs['area'],
        'zone': inputs['zone'],
        'location_hub': inputs['location_hub'],
        'property_type': inputs['property_type'],
        'ownership': inputs['ownership'],
        'floor_no': floor_to_int_list(inputs['floor_no']),
        'electric_charge_included': inputs['electric_charge_included'],
        'water_charge_included': inputs['water_charge_included'],
        'possession_status': inputs['possession_status'],
        'posted_by': inputs['posted_by'],
        'negotiable': inputs['negotiable'],
        'brokerage': inputs['brokerage']
    }
    
    # Create one-hot encoded columns
    for feature, value in categorical_mappings.items():
        if feature == 'floor_no':
            col_name = f"floor_no_{value}"
        elif feature in ['electric_charge_included', 'water_charge_included', 
                         'possession_status', 'negotiable', 'brokerage']:
            col_name = f"{feature}_{'yes' if value.lower() == 'yes' else 'no'}"
        else:
            col_name = f"{feature}_{value.lower().replace(' ', '_').replace('/', '_')}"
        
        if col_name in input_df.columns:
            input_df[col_name] = 1
    
    # Scale numerical features
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    return input_df

def predict_rent(inputs):
    try:
        processed_input = process_user_input(inputs)
        prediction = model.predict(processed_input)
        return np.expm1(prediction[0])  # Reverse log transformation
    except Exception as e:
        return f"Error making prediction: {str(e)}"

# Main app
def main():
    # Header
    st.title("🏢 Commercial Property Rent Predictor")
    st.markdown("Fill in the details below to predict the monthly rent for a commercial property in Nagpur.")
    
    if model is None:
        st.error("Model components not loaded. Please ensure 'mc.pkl', 'sc.pkl', and 'fc.pkl' are in the same directory.")
        return
    
    # Define options for categorical features
    listing_options = ['showroom for rent', 'shop for rent', 'bare shell office space', 
                     'ready to use office space', 'commercial property', 'werehouse', 'godown for rent']
    
    area_options = ['manewada', 'jaitala', 'besa', 'omkar nagar', 'itwari', 'hingna', 
                   'sitabuldi', 'mahal', 'kharbi', 'mihan', 'pratap nagar', 'ramdaspeth', 
                   'dharampeth', 'gandhibag', 'chatrapati nagar', 'nandanwan', 'sadar']
    
    zone_options = ['south', 'west', 'east', 'north']
    
    location_hub_options = ['commercial project', 'others', 'retail complex/building', 
                           'market/high street', 'business park', 'it park', 'residential']
    
    property_type_options = ['showroom', 'shop', 'bare shell office', 'ready to use office', 
                           'commercial property', 'werehouse', 'godown']
    
    ownership_options = ['freehold', 'leasehold', 'cooperative society', 'power_of_attorney']
    
    possession_options = ['ready to move', 'Under Construction']
    
    posted_by_options = ['owner', 'housing expert', 'broker']
    
    amenities_options = ['parking', 'vastu', 'lift', 'cabin', 'meeting room', 
                        'CCTV', 'water storage', 'power backup', 'security', 
                        'fire safety', 'internet', 'pantry', 'food court', 
                        'fire extinguishers', 'reception area', 'furnishing']
    
    # Create form
    with st.form("prediction_form"):
        st.header("Property Details")
        
        # Two columns for layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Listing title
            listing_title = st.selectbox("Listing Title", listing_options)
            
            # Area
            area = st.selectbox("Area", area_options)
            
            # Zone
            zone = st.selectbox("Zone", zone_options)
            
            # Location Hub
            location_hub = st.selectbox("Location Hub", location_hub_options)
            
            # Property Type
            property_type = st.selectbox("Property Type", property_type_options)
            
            # Ownership
            ownership = st.selectbox("Ownership", ownership_options)
            
            # Size in sqft
            size_in_sqft = st.number_input("Size (sqft)", min_value=100, max_value=100000, value=1000)
            
            # Carpet area sqft
            carpet_area_sqft = st.number_input("Carpet Area (sqft)", min_value=50, max_value=90000, value=800)
            
            # Private washroom
            private_washroom = st.slider("Private Washrooms", min_value=0, max_value=3, value=1)
            
            # Public washroom
            public_washroom = st.slider("Public Washrooms", min_value=0, max_value=3, value=1)
            
            # Floor number
            floor_no = st.text_input("Floor Number(s)", value="1", 
                                      help="Enter floor numbers separated by commas (e.g., '1' or '0,1,2')")
            
            # Total floors
            total_floors = st.number_input("Total Floors", min_value=1, max_value=50, value=3)
        
        with col2:
            # Amenities
            amenities = st.multiselect("Amenities", amenities_options, default=[])
            
            # Electric charge included
            electric_charge_included = st.radio("Electric Charge Included", ['yes', 'no'], index=1)
            
            # Water charge included
            water_charge_included = st.radio("Water Charge Included", ['yes', 'no'], index=0)
            
            # Property age
            property_age = st.number_input("Property Age (years)", min_value=0, max_value=50, value=5)
            
            # Possession status
            possession_status = st.radio("Possession Status", possession_options)
            
            # Posted by
            posted_by = st.radio("Posted By", posted_by_options)
            
            # Lock in period
            lock_in_period_in_months = st.number_input("Lock In Period (months)", min_value=0, max_value=60, value=6)
            
            # Expected rent increase
            expected_rent_increase = st.number_input("Expected Yearly Rent Increase", 
                                                    min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                                                    format="%.2f")
            
            # Negotiable
            negotiable = st.radio("Negotiable", ['yes', 'no'], index=0)
            
            # Brokerage
            brokerage = st.radio("Brokerage", ['yes', 'no'], index=0)
        
        # Submit button
        submit_button = st.form_submit_button(label="Predict Rent Price")
    
    # Process prediction when form is submitted
    if submit_button:
        # Prepare input dictionary
        inputs = {
            'listing litle': listing_title,
            'area': area,
            'zone': zone,
            'location_hub': location_hub,
            'property_type': property_type,
            'ownership': ownership,
            'size_in_sqft': size_in_sqft,
            'carpet_area_sqft': carpet_area_sqft,
            'private_washroom': private_washroom,
            'public_washroom': public_washroom,
            'floor_no': floor_no,
            'total_floors': total_floors,
            'amenities': amenities,
            'electric_charge_included': electric_charge_included,
            'water_charge_included': water_charge_included,
            'property_age': property_age,
            'possession_status': possession_status,
            'posted_by': posted_by,
            'lock_in_period_in_months': lock_in_period_in_months,
            'expected rent increases yearly': expected_rent_increase,
            'negotiable': negotiable,
            'brokerage': brokerage
        }
        
        # Make prediction
        with st.spinner("Calculating prediction..."):
            prediction = predict_rent(inputs)
        
        # Display prediction
        st.header("Prediction Result")
        
        if isinstance(prediction, str):
            st.error(prediction)
        else:
            st.success(f"Predicted Monthly Rent: ₹{prediction:,.2f}")
            
            # Display additional information
            st.subheader("Property Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Property Type", property_type)
                st.metric("Area", area)
                st.metric("Zone", zone)
                
            with col2:
                st.metric("Size", f"{size_in_sqft:,} sqft")
                st.metric("Carpet Area", f"{carpet_area_sqft:,} sqft")
                st.metric("Property Age", f"{property_age} years")
                
            with col3:
                st.metric("Location Hub", location_hub)
                st.metric("Possession", possession_status)
                st.metric("Posted By", posted_by)
            
            # Display selected amenities
            if amenities:
                st.subheader("Selected Amenities")
                amenities_str = ", ".join(amenities)
                st.info(amenities_str)

if __name__ == "__main__":
    main()
