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

# Custom CSS for white text, dark theme, and compact layout
st.markdown("""
<style>
    /* Set default text color to white and background to dark */
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    html, body, .stMarkdown, h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: white !important;
    }
    /* Make widgets more compact and ensure text is white */
    .stSelectbox > div > div > div { color: white; }
    .stNumberInput > div > div > input { color: white; }
    .stMultiSelect > div > div > div { color: white; }
    .element-container { margin-bottom: 0.5rem; }
    .stForm { border: 0px; padding: 0rem; }
    /* Style expander header */
    .streamlit-expanderHeader {
        background-color: #262730;
        border-radius: 5px;
    }
    /* Style multiselect dropdown */
    .stMultiSelect div[data-baseweb="select"] span {
        color: white;
    }
    /* Custom styling for price display */
    .price-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .price-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .price-value {
        font-size: 28px;
        font-weight: bold;
    }
    .price-change {
        font-size: 16px;
        margin-top: 5px;
    }
    .positive-change {
        color: #4ade80;
    }
    .negative-change {
        color: #f87171;
    }
</style>
""", unsafe_allow_html=True)

# Floor weightage system (percentage adjustments)
FLOOR_WEIGHTAGE = {
    0: 0.0,    # Ground floor - base price
    1: 2.0,    # 1st floor - 2% increase
    2: 4.0,    # 2nd floor - 4% increase
    3: 6.0,    # 3rd floor - 6% increase
    4: 8.0,    # 4th floor - 8% increase
    5: 10.0,   # 5th floor - 10% increase
    6: 12.0,   # 6th floor - 12% increase
    7: 14.0,   # 7th floor - 14% increase
    8: 16.0,   # 8th floor - 16% increase
    9: 18.0,   # 9th floor - 18% increase
    10: 20.0   # 10th floor - 20% increase
}

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

# Calculate floor-adjusted rent
def calculate_floor_adjusted_rent(base_rent, selected_floors):
    if not selected_floors:
        return base_rent, 0.0
    
    # Calculate average weightage for selected floors
    total_weightage = sum(FLOOR_WEIGHTAGE.get(int(floor), 0) for floor in selected_floors)
    avg_weightage = total_weightage / len(selected_floors)
    
    # Apply weightage to base rent
    adjusted_rent = base_rent * (1 + avg_weightage / 100)
    
    return adjusted_rent, avg_weightage

# Main function to run the app
def main():
    model, scaler, feature_names = load_model_components()
    
    if model is None or scaler is None or feature_names is None:
        st.error("Unable to load model components. Please check your files.")
        return
    
    st.markdown('<h1 style="text-align: center;">🏢 Commercial Property Rent Predictor</h1>', unsafe_allow_html=True)
    
    # Display floor weightage information
    with st.expander("📊 Floor Premium Rates", expanded=False):
        st.write("Rent adjustments based on floor selection:")
        weightage_df = pd.DataFrame(list(FLOOR_WEIGHTAGE.items()), columns=['Floor', 'Premium (%)'])
        weightage_df['Floor'] = weightage_df['Floor'].apply(lambda x: f"Floor {x}")
        st.dataframe(weightage_df, hide_index=True, use_container_width=True)
    
    with st.form("prediction_form"):
        # --- Compact Input Grid ---
        col1, col2, col3 = st.columns(3)
        with col1:
            property_type = st.selectbox("Property Type", ['showroom', 'shop', 'bare shell office', 'ready to use office', 'commercial property', 'werehouse', 'godown'], index=0)
            size_sqft = st.number_input("Size (sqft)", min_value=100, max_value=100000, value=1000, step=50)
            area = st.selectbox("Area", ['manewada', 'jaitala', 'besa', 'omkar nagar', 'itwari', 'hingna', 'sitabuldi', 'mahal', 'kharbi', 'mihan', 'pratap nagar', 'ramdaspeth', 'dharampeth', 'gandhibag', 'chatrapati nagar', 'nandanwan', 'sadar', 'dighori', 'somalwada', 'ganeshpeth colony', 'mhalgi nagar', 'sakkardara', 'babulban', 'manish nagar', 'dhantoli', 'khamla', 'laxminagar', 'ajni', 'wathoda', 'hulkeshwar', 'pardi', 'new indora', 'civil lines', 'gadhibag', 'bagadganj', 'swawlambi nagar', 'manawada', 'trimurti nagar', 'lakadganj', 'shivaji nagar'], index=0)
        with col2:
            carpet_area = st.number_input("Carpet Area (sqft)", min_value=100, max_value=100000, value=800, step=50)
            zone = st.selectbox("Zone", ['south', 'west', 'east', 'north'], index=0)
            location_hub = st.selectbox("Location Hub", ['commercial project', 'others', 'retail complex/building', 'market/high street', 'business park', 'it park', 'residential'], index=0)
        with col3:
            ownership = st.selectbox("Ownership", ['freehold', 'leasehold', 'cooperative society', 'power_of_attorney'], index=0)
            total_floors = st.selectbox("Total Floors", ['3 floors', '1 floor', '2 floors', '4 floors', '5 floors', '8 floors', '7 floors', '6 floors', '15 floors', '9 floors', '10 floors'], index=0)
            # --- Floor Selection with Multiselect Dropdown ---
            floor_options = [f"Floor {i}" for i in range(0, 11)]
            selected_floors = st.multiselect("Select Available Floors", floor_options, default=["Floor 0"])

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            private_washroom = st.number_input("Private Washrooms", min_value=0, max_value=20, value=1)
        with col_b:
            public_washroom = st.number_input("Public Washrooms", min_value=0, max_value=20, value=1)
        with col_c:
            property_age = st.number_input("Property Age (years)", min_value=0, max_value=100, value=5)

        # --- Expanders for less critical details ---
        with st.expander("Amenities & Charges"):
            amenities_options = ['parking', 'vastu', 'lift', 'cabin', 'meeting room', 'dg and ups', 'water storage', 'staircase', 'security', 'cctv', 'power backup', 'reception area', 'pantry', 'fire extinguishers', 'fire safety', 'oxygen duct', 'food court', 'furnishing', 'internet', 'fire sensors']
            selected_amenities = st.multiselect("Select Amenities", amenities_options)
            electric_charge = st.selectbox("Electric Charge Included", ['yes', 'no'], index=0)
            water_charge = st.selectbox("Water Charge Included", ['yes', 'no'], index=0)

        with st.expander("Other Details"):
            possession_status = st.selectbox("Possession Status", ['ready to move', 'Under Construction'], index=0)
            posted_by = st.selectbox("Posted By", ['owner', 'housing expert', 'broker'], index=0)
            lock_in_period_str = st.selectbox("Lock-in Period", ['2 months', '6 months', '12 months', '3 months', '1 month', '11 months', '4 months', '10 months', '6  months', '8  months', '4  months', '36 months'], index=0)
            expected_rent_increase_str = st.selectbox("Yearly Rent Increase", ['0.05', '0.10'], index=0)
            negotiable = st.selectbox("Negotiable", ['yes', 'no'], index=0)
            brokerage = st.selectbox("Brokerage", ['yes', 'no'], index=0)
        
        # --- Submit Button ---
        predict_button = st.form_submit_button("Predict Rent Price", use_container_width=True)
        
        if predict_button:
            # Process the selected floors from multiselect
            floor_numbers = [floor.replace("Floor ", "") for floor in selected_floors]
            floor_no_str = ",".join(sorted(floor_numbers))
            
            lock_in_period = int(re.sub(r'\D', '', lock_in_period_str))
            expected_rent_increase = float(expected_rent_increase_str)
            
            user_data = {
                'listing litle': property_type, 'city': 'nagpur', 'area': area, 'zone': zone,
                'location_hub': location_hub, 'property_type': property_type, 'ownership': ownership,
                'size_in_sqft': size_sqft, 'carpet_area_sqft': carpet_area,
                'private_washroom': private_washroom, 'public_washroom': public_washroom,
                'floor_no': floor_no_str, 'total_floors': total_floors,
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
                base_prediction = np.expm1(prediction_log)
                
                # Calculate floor-adjusted rent
                adjusted_rent, avg_weightage = calculate_floor_adjusted_rent(base_prediction, floor_numbers)
                
                st.session_state.base_prediction = base_prediction
                st.session_state.adjusted_prediction = adjusted_rent
                st.session_state.avg_weightage = avg_weightage
                st.session_state.user_data = user_data
                st.session_state.processed_df = processed_df
                st.session_state.selected_floors = selected_floors
                st.success("Prediction successful! See the results below.")

            except Exception as e:
                st.error(f"Error making prediction: {e}")

    # --- Prediction Results Section ---
    if 'base_prediction' in st.session_state:
        st.markdown("---")
        st.markdown('<h2>Prediction Results</h2>', unsafe_allow_html=True)
        
        base_prediction = st.session_state.base_prediction
        adjusted_prediction = st.session_state.adjusted_prediction
        avg_weightage = st.session_state.avg_weightage
        user_data = st.session_state.user_data
        selected_floors = st.session_state.selected_floors
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Base Price Display
            st.markdown('<div class="price-container">', unsafe_allow_html=True)
            st.markdown('<div class="price-label">Base Rent Price (Ground Floor)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="price-value">₹{base_prediction:.2f}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Adjusted Price Display
            st.markdown('<div class="price-container" style="background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);">', unsafe_allow_html=True)
            st.markdown('<div class="price-label">Estimated Rent Price (Floor Adjusted)</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="price-value">₹{adjusted_prediction:.2f}</div>', unsafe_allow_html=True)
            
            # Show percentage change
            if avg_weightage > 0:
                st.markdown(f'<div class="price-change positive-change">+{avg_weightage:.1f}% Floor Premium</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="price-change">No Floor Premium</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('<h4>Property Summary</h4>', unsafe_allow_html=True)
            st.write(f"**Property Type:** {user_data['property_type'].title()}")
            st.write(f"**Size:** {user_data['size_in_sqft']} sqft")
            st.write(f"**Area:** {user_data['area'].title()}")
            # Display selected floors
            floors_list = user_data['floor_no'].split(',')
            if len(floors_list) == 1:
                st.write(f"**Floor:** Floor {floors_list[0]}")
            else:
                st.write(f"**Floors:** {', '.join([f'Floor {f}' for f in floors_list])}")
        
        with col2:
            st.markdown('<h4>Price Comparison</h4>', unsafe_allow_html=True)
            lower_bound = adjusted_prediction * 0.85
            upper_bound = adjusted_prediction * 1.15
            st.write(f"**Fair Range:** ₹{lower_bound:.2f} - ₹{upper_bound:.2f}")
            comparison_price = st.number_input("Enter Listed Price", min_value=0.0, value=float(adjusted_prediction), step=1000.0)
            if comparison_price < lower_bound:
                st.warning("Listed price is **BELOW** fair range.")
            elif comparison_price > upper_bound:
                st.warning("Listed price is **ABOVE** fair range.")
            else:
                st.success("Listed price is **FAIR**.")
            
            st.markdown('<h4>Future Projection</h4>', unsafe_allow_html=True)
            years = st.slider("Years", min_value=1, max_value=10, value=5)
            growth_rate = st.slider("Growth (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.5)
            projected_price = adjusted_prediction * ((1 + growth_rate/100) ** years)
            st.write(f"**Rent in {years} years:** ₹{projected_price:.2f}")
            
            # --- MODIFIED PLOTTING SECTION ---
            fig, ax = plt.subplots(figsize=(10, 5))
            years_range = np.arange(0, years + 1)
            prices = [adjusted_prediction * ((1 + growth_rate/100) ** y) for y in years_range]
            
            # Set plot background to white
            ax.set_facecolor('#FFFFFF')
            fig.patch.set_facecolor('#FFFFFF')
            
            # Plot line with a visible color
            ax.plot(years_range, prices, marker='o', linestyle='-', color='#1f77b4')
            
            # Set all text and grid elements to black
            ax.set_title(f'Rent Projection ({growth_rate}% Growth)', color='black')
            ax.set_xlabel('Years', color='black')
            ax.set_ylabel('Rent Price (₹)', color='black')
            ax.tick_params(colors='black')
            ax.grid(True, linestyle='--', color='black', alpha=0.3)
            
            st.pyplot(fig)
            
            # Floor Impact Analysis
            st.markdown('<h4>Floor Impact Analysis</h4>', unsafe_allow_html=True)
            floor_impact_data = []
            for floor in selected_floors:
                floor_num = int(floor.replace("Floor ", ""))
                premium = FLOOR_WEIGHTAGE.get(floor_num, 0)
                floor_price = base_prediction * (1 + premium / 100)
                floor_impact_data.append({
                    'Floor': floor,
                    'Premium (%)': premium,
                    'Price': f"₹{floor_price:.2f}"
                })
            
            if floor_impact_data:
                impact_df = pd.DataFrame(floor_impact_data)
                st.dataframe(impact_df, hide_index=True, use_container_width=True)

# Run the app
if __name__ == "__main__":
    main()
