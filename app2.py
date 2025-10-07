import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Commercial Property Rent Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and black text
st.markdown("""
<style>
    /* Set default text color to black */
    html, body, .stApp, .stMarkdown {
        color: black;
    }
    .main-header {
        font-size: 2.5rem;
        color: black;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: black;
        margin-bottom: 1rem;
    }
    .prediction-result {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model, scaler, and feature names
@st.cache_resource(show_spinner=True)
def load_model_components():
    try:
        model = joblib.load('mc.pkl')
        scaler = joblib.load('sc.pkl')
        feature_names = joblib.load('fc.pkl')
        return model, scaler, feature_names
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None

# Preprocess input data to match model requirements
def preprocess_input(user_data, feature_names, scaler):
    df = pd.DataFrame([user_data])
    
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
        return None
    
    df['floor_no'] = df['floor_no'].apply(floor_to_int_list)
    
    def total_floors_to_int(floor_str):
        try:
            return int(str(floor_str).lower().replace('floors', '').replace('floor', '').strip())
        except:
            return None
    
    df['total_floors'] = df['total_floors'].apply(total_floors_to_int)
    
    def size_to_int(size_str):
        try:
            return int(str(size_str).lower().replace('sqft','').replace('sq.ft','').strip())
        except:
            return None
    
    df['size_in_sqft'] = df['size_in_sqft'].apply(size_to_int)
    df['carpet_area_sqft'] = df['carpet_area_sqft'].apply(size_to_int)
    
    def extract_amenities_list(text):
        text = str(text).lower()
        text = re.sub(r'\(\d+\)', '', text)
        amenities = [x.strip() for x in text.split(',') if x.strip() != '']
        return amenities
    
    all_amenities = ['parking', 'vastu', 'lift', 'cabin', 'meeting room', 'dg and ups', 
                    'water storage', 'staircase', 'security', 'cctv', 'power backup', 
                    'reception area', 'pantry', 'fire extinguishers', 'fire safety', 
                    'oxygen duct', 'food court', 'furnishing', 'internet', 'fire sensors']
    
    for amenity in all_amenities:
        amenity_col = amenity.replace(' ', '_')
        df[amenity_col] = df['amenities_count'].apply(
            lambda x: 1 if amenity in extract_amenities_list(x) else 0
        )
    
    df['property_age'] = df['property_age'].astype(int)
    df['lock_in_period_in_months'] = df['lock in period'].str.replace('months', '', regex=False) \
                                     .str.replace('month', '', regex=False) \
                                     .str.strip() \
                                     .fillna(0) \
                                     .astype(int)
    
    categorical_features = ['listing litle', 'city', 'area', 'zone', 'location_hub',
                           'property_type', 'ownership', 'floor_no',
                           'electric_charge_included', 'water_charge_included',
                           'possession_status', 'posted_by', 'negotiable', 'brokerage']
    
    for feature in categorical_features:
        if feature in df.columns:
            df = pd.get_dummies(df, columns=[feature])
    
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_names]
    
    numerical_cols = ['size_in_sqft', 'carpet_area_sqft', 'private_washroom',
                     'public_washroom', 'total_floors', 'property_age',
                     'expected rent increases yearly', 'fire_extinguishers',
                     'food_court', 'cabin', 'lift', '0', 'water_storage', 'dg',
                     'fire_safety', 'security', 'cctv', 'oxygen_duct', 'furnishing', 'vastu',
                     'reception_area', 'internet', 'water_supply', 'fire_sensors',
                     'power_backup', 'dg_and_ups', 'parking', 'pantry',
                     'lock_in_period_in_months']
    
    numerical_cols_present = [col for col in numerical_cols if col in df.columns]
    
    if numerical_cols_present:
        df[numerical_cols_present] = scaler.transform(df[numerical_cols_present])
    
    return df

# Function to display options and get user selection
def display_options(options, title):
    st.write(f"**{title}:**")
    selected_option = st.selectbox(f"Select {title.lower()}", options)
    return selected_option

# Main function to run the app
def main():
    # Load model components
    model, scaler, feature_names = load_model_components()
    
    if model is None or scaler is None or feature_names is None:
        st.error("Unable to load model components. Please check your files.")
        return
    
    # App header
    st.markdown('<h1 class="main-header">🏢 Commercial Property Rent Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Enter property details to predict the rental price</p>', unsafe_allow_html=True)
    
    # --- Property Details Form ---
    st.markdown('<h2 class="sub-header">Property Information</h2>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Basic Information")
            property_types = ['showroom', 'shop', 'bare shell office', 'ready to use office', 
                            'commercial property', 'werehouse', 'godown']
            property_type = display_options(property_types, "Property Type")
            size_sqft = st.number_input("Size in sqft", min_value=100, max_value=100000, value=1000, step=50)
            carpet_area = st.number_input("Carpet Area in sqft", min_value=100, max_value=100000, value=800, step=50)
            
            st.write("### Location Information")
            areas = ['manewada', 'jaitala', 'besa', 'omkar nagar', 'itwari', 'hingna', 
                    'sitabuldi', 'mahal', 'kharbi', 'mihan', 'pratap nagar', 'ramdaspeth', 
                    'dharampeth', 'gandhibag', 'chatrapati nagar', 'nandanwan', 'sadar', 
                    'dighori', 'somalwada', 'ganeshpeth colony', 'mhalgi nagar', 'sakkardara', 
                    'babulban', 'manish nagar', 'dhantoli', 'khamla', 'laxminagar', 'ajni', 
                    'wathoda', 'hulkeshwar', 'pardi', 'new indora', 'civil lines', 'gadhibag', 
                    'bagadganj', 'swawlambi nagar', 'manawada', 'trimurti nagar', 'lakadganj', 'shivaji nagar']
            area = display_options(areas, "Area")
            zones = ['south', 'west', 'east', 'north']
            zone = display_options(zones, "Zone")
            location_hubs = ['commercial project', 'others', 'retail complex/building', 
                            'market/high street', 'business park', 'it park', 'residential']
            location_hub = display_options(location_hubs, "Location Hub")
            ownerships = ['freehold', 'leasehold', 'cooperative society', 'power_of_attorney']
            ownership = display_options(ownerships, "Ownership Type")
        
        with col2:
            st.write("### Property Features")
            private_washroom = st.number_input("Number of private washrooms", min_value=0, max_value=20, value=1)
            public_washroom = st.number_input("Number of public washrooms", min_value=0, max_value=20, value=1)
            floor_options = ['ground floor', '1 floor', '2 floor', '1, 2,3 floors', 
                            'ground floor,1 floor', '1,2,3 floors', '1,2 floors', 
                            '1,2,3,4,GF', '1 , GF floor', '8 floor', '3 floor']
            floor_no = display_options(floor_options, "Floor Number")
            total_floors_options = ['3 floors', '1 floor', '2 floors', '4 floors', 
                                   '5 floors', '8 floors', '7 floors', '6 floors', 
                                   '15 floors', '9 floors', '10 floors']
            total_floors = display_options(total_floors_options, "Total Floors in Building")
            
            st.write("### Amenities")
            amenities_options = ['parking', 'vastu', 'lift', 'cabin', 'meeting room', 'dg and ups', 
                                'water storage', 'staircase', 'security', 'cctv', 'power backup', 
                                'reception area', 'pantry', 'fire extinguishers', 'fire safety', 
                                'oxygen duct', 'food court', 'furnishing', 'internet', 'fire sensors']
            selected_amenities = st.multiselect("Select Amenities", amenities_options)
            
            st.write("### Other Details")
            yes_no_options = ['yes', 'no']
            electric_charge = display_options(yes_no_options, "Electric charge included")
            water_charge = display_options(yes_no_options, "Water charge included")
            property_age = st.number_input("Property age in years", min_value=0, max_value=100, value=5)
            possession_statuses = ['ready to move', 'Under Construction']
            possession_status = display_options(possession_statuses, "Possession status")
            posted_by_options = ['owner', 'housing expert', 'broker']
            posted_by = display_options(posted_by_options, "Posted by")
            lock_in_period_options = ['2 months', '6 months', '12 months', '3 months', '1 month', 
                                     '11 months', '4 months', '10 months', '6  months', '8  months', 
                                     '4  months', '36 months']
            lock_in_period_str = display_options(lock_in_period_options, "Lock-in period")
            lock_in_period = int(re.sub(r'\D', '', lock_in_period_str))
            expected_rent_increase_options = ['0.05', '0.10']
            expected_rent_increase_str = display_options(expected_rent_increase_options, "Expected yearly rent increase")
            expected_rent_increase = float(expected_rent_increase_str)
            negotiable = display_options(yes_no_options, "Negotiable")
            brokerage = display_options(yes_no_options, "Brokerage")
        
        # Prediction button
        st.markdown("---")
        predict_button = st.button("Predict Rent Price", use_container_width=True, type="primary")
        
        if predict_button:
            user_data = {
                'listing litle': property_type, 'city': 'nagpur', 'area': area, 'zone': zone,
                'location_hub': location_hub, 'property_type': property_type, 'ownership': ownership,
                'size_in_sqft': size_sqft, 'carpet_area_sqft': carpet_area,
                'private_washroom': private_washroom, 'public_washroom': public_washroom,
                'floor_no': floor_no, 'total_floors': total_floors,
                'amenities_count': ', '.join(selected_amenities),
                'electric_charge_included': electric_charge, 'water_charge_included': water_charge,
                'property_age': property_age, 'possession_status': possession_status,
                'posted_by': posted_by, 'lock in period': f"{lock_in_period} months",
                'expected rent increases yearly': expected_rent_increase,
                'negotiable': negotiable, 'brokerage': brokerage
            }
            
            processed_df = preprocess_input(user_data, feature_names, scaler)
            
            try:
                prediction_log = model.predict(processed_df)[0]
                prediction = np.expm1(prediction_log)
                
                st.session_state.prediction = prediction
                st.session_state.user_data = user_data
                st.session_state.processed_df = processed_df
                st.success("Prediction successful! See the results below.")

            except Exception as e:
                st.error(f"Error making prediction: {e}")

    # --- Prediction Results Section ---
    if 'prediction' in st.session_state:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Prediction Results</h2>', unsafe_allow_html=True)
        
        prediction = st.session_state.prediction
        user_data = st.session_state.user_data
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prediction-result">', unsafe_allow_html=True)
            st.markdown(f'<h3>Estimated Rent Price</h3>', unsafe_allow_html=True)
            st.markdown(f'<h1>₹{prediction:.2f}</h1>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<h3>Property Summary</h3>', unsafe_allow_html=True)
            st.write(f"**Property Type:** {user_data['property_type'].title()}")
            st.write(f"**Size:** {user_data['size_in_sqft']} sqft")
            st.write(f"**Carpet Area:** {user_data['carpet_area_sqft']} sqft")
            st.write(f"**Area:** {user_data['area'].title()}")
            st.write(f"**Zone:** {user_data['zone'].title()}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<h3>Price Comparison</h3>', unsafe_allow_html=True)
            lower_bound = prediction * 0.85
            upper_bound = prediction * 1.15
            st.write(f"**Fair Price Range:** ₹{lower_bound:.2f} - ₹{upper_bound:.2f}")
            comparison_price = st.number_input("Enter Listed Price for Comparison", 
                                             min_value=0.0, value=float(prediction), step=1000.0)
            if comparison_price < lower_bound:
                st.warning("The listed price is **below** the estimated fair price range.")
            elif comparison_price > upper_bound:
                st.warning("The listed price is **above** the estimated fair price range.")
            else:
                st.success("The listed price is **within** the estimated fair price range.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown('<h3>Future Rent Projection</h3>', unsafe_allow_html=True)
            years = st.slider("Years for Projection", min_value=1, max_value=10, value=5)
            growth_rate = st.slider("Annual Growth Rate (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
            projected_price = prediction * ((1 + growth_rate/100) ** years)
            st.write(f"**Projected Rent in {years} years:** ₹{projected_price:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            years_range = np.arange(0, years + 1)
            prices = [prediction * ((1 + growth_rate/100) ** y) for y in years_range]
            ax.plot(years_range, prices, marker='o', linestyle='-', color='#1f77b4')
            ax.set_title(f'Rent Projection Over {years} Years at {growth_rate}% Annual Growth')
            ax.set_xlabel('Years')
            ax.set_ylabel('Rent Price (₹)')
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)

    # --- Feature Analysis Section ---
    if 'processed_df' in st.session_state:
        st.markdown("---")
        st.markdown('<h2 class="sub-header">Feature Analysis</h2>', unsafe_allow_html=True)
        
        processed_df = st.session_state.processed_df
        
        if hasattr(model, 'feature_importances_'):
            st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
            st.markdown('<h3>Feature Importance</h3>', unsafe_allow_html=True)
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            top_features = 15
            top_indices = indices[:top_features]
            feature_importance_df = pd.DataFrame({
                'Feature': [feature_names[i] for i in top_indices],
                'Importance': importances[top_indices]
            })
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax)
            ax.set_title(f'Top {top_features} Feature Importances')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
        st.markdown('<h3>Current Property Feature Values</h3>', unsafe_allow_html=True)
        non_zero_features = processed_df.loc[:, (processed_df != 0).any(axis=0)]
        numerical_features = non_zero_features.select_dtypes(include=['number']).columns.tolist()
        categorical_features = non_zero_features.select_dtypes(exclude=['number']).columns.tolist()
        
        if numerical_features:
            st.write("**Numerical Features:**")
            num_features_df = non_zero_features[numerical_features].T
            num_features_df.columns = ['Value']
            st.dataframe(num_features_df)
        
        if categorical_features:
            st.write("**Categorical Features:**")
            cat_features_df = non_zero_features[categorical_features].T
            cat_features_df.columns = ['Value']
            st.dataframe(cat_features_df)
        st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
