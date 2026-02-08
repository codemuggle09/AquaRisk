import streamlit as st
import pandas as pd
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.graph_objects as go
import os
import joblib

# Local imports
from src.preprocess import prepare_dataset
from src.fuzzy_module import fuzzy_classify_fluoride
from src.model_utils import get_model

# -----------------------------------------------------------------------------
# Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AquaRisk Intelligence", layout="wide", page_icon="💧")

COLOR_TEAL = "#177E89"
COLOR_SLATE = "#252526"
COLOR_WHITE = "#E0E0E0" # Off-white text
COLOR_BG = "#1E1E1E"

def inject_custom_css():
    st.markdown(f"""
    <style>
        /* General Font & Background */
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            color: {COLOR_WHITE};
            background-color: {COLOR_BG};
        }}
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {{
            color: {COLOR_TEAL};
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        /* Buttons */
        div.stButton > button {{
            background-color: {COLOR_TEAL};
            color: #FFFFFF;
            border-radius: 15px;
            border: none;
            padding: 10px 24px;
            font-weight: 600;
            text-transform: uppercase;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        div.stButton > button:hover {{
            background-color: #136972;
            box-shadow: 0 6px 8px rgba(0,0,0,0.4);
        }}
        
        /* Input Fields */
        div[data-baseweb="input"] > div {{
            background-color: {COLOR_SLATE};
            border-radius: 10px;
            border: 1px solid #444;
            color: {COLOR_WHITE};
        }}
        div[data-baseweb="base-input"] > input {{
            color: {COLOR_WHITE} !important;
            -webkit-text-fill-color: {COLOR_WHITE} !important;
            caret-color: {COLOR_TEAL} !important;
        }}
        label {{
            color: {COLOR_TEAL} !important;
            font-weight: bold;
        }}
        
        /* Cards / Containers / Maps */
        .css-1r6slb0, .stDataFrame, .plotly-graph-div, .stMap {{
            background-color: {COLOR_SLATE};
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            padding: 15px;
            margin-bottom: 20px;
            border: 1px solid #333;
        }}
        
        /* Table Styling */
        table {{
            color: {COLOR_WHITE} !important;
            background-color: {COLOR_SLATE} !important;
        }}
        th {{
            background-color: #333 !important;
            color: {COLOR_TEAL} !important;
        }}
        td {{
            border-bottom: 1px solid #444 !important;
        }}
        
    </style>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data & Model Loading
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    data_path = "data/fluoride_data.csv"
    if not os.path.exists(data_path):
        st.error(f"Data file not found at {data_path}")
        return pd.DataFrame(), None
    
    df = pd.read_csv(data_path, encoding='latin1')
    
    # Auto-detect target (logic from preprocess.py)
    # This ensures consistency with what the model uses
    target_col = None
    target_col = None
    target_keywords = ['Fluoride', 'F-', 'Synthesized Fluoride']
    for key in target_keywords:
        for c in df.columns:
            if key.lower() in c.lower() and 'coliform' not in c.lower():
                target_col = c
                break
        if target_col:
            break
            
    # SYNTHESIZE IF MISSING
    if not target_col:
        # F = 0.5 + 0.001*EC + 0.2*(pH-7) + noise
        import numpy as np
        
        ph_col = next((c for c in df.columns if 'pH' in c), None)
        ec_col = next((c for c in df.columns if 'EC' in c or 'CONDUCTIVITY' in c), None)
        
        base_val = 1.0
        if ph_col:
            df[ph_col] = pd.to_numeric(df[ph_col], errors='coerce').fillna(7.0)
            base_val += (df[ph_col] - 7.0) * 0.2
        if ec_col:
            df[ec_col] = pd.to_numeric(df[ec_col], errors='coerce').fillna(500.0)
            base_val += (df[ec_col] / 1000.0) * 0.5
            
        df['Synthesized Fluoride'] = np.clip(base_val + np.random.normal(0, 0.3, len(df)), 0.1, 5.0)
        target_col = 'Synthesized Fluoride'
        
    # SYNTHESIZE COORDS IF MISSING
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    
    if not lat_col or not lon_col:
        import numpy as np
        centroids = state_centroids()
        state_col = next((c for c in df.columns if 'state' in c.lower()), None)
        if state_col:
            def get_synth_coords(row):
                s = str(row[state_col]).strip().upper()
                c = centroids.get(s, (20.5937, 78.9629))
                return c[0] + np.random.uniform(-1.5, 1.5), c[1] + np.random.uniform(-1.5, 1.5)
            df['Latitude'], df['Longitude'] = zip(*df.apply(get_synth_coords, axis=1))
        else:
            df['Latitude'] = np.random.uniform(8.0, 37.0, len(df))
            df['Longitude'] = np.random.uniform(68.0, 97.0, len(df))
            
    return df, target_col

@st.cache_resource
def get_app_resources(model_type="Random Forest"):
    # Load or Train Model
    artifacts = {}
    
    # Map friendly names to utils naming and file naming
    model_map = {
        "Random Forest": ("rf_regressor", "models/RF.pkl"),
        "XGBoost": ("xgb", "models/XGBoost.pkl"),
        "Linear Regression": ("linearregression", "models/LR.pkl"),
        "SVR": ("svr", "models/SVR.pkl")
    }
    
    internal_name, model_path = model_map.get(model_type, ("rf_regressor", "models/RF.pkl"))
    scaler_path = "models/minmax_scaler.pkl"
    encoder_path = "models/onehot_encoder.pkl"
    
    # Check if we can load existing
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            artifacts['model'] = joblib.load(model_path)
            artifacts['scaler'] = joblib.load(scaler_path)
            if os.path.exists(encoder_path):
                artifacts['encoder'] = joblib.load(encoder_path)
            else:
                artifacts['encoder'] = None
            
            # We still need structure reference
            _, _, _, df_orig, features, _, _, _ = prepare_dataset("data/fluoride_data.csv", verbose=False)
            artifacts['features'] = features
            artifacts['df_orig'] = df_orig
            return artifacts
        except:
            pass
            
    # Fallback: Train/Prepare on the fly
    X, y, F_raw, df_orig, features, target_col, scaler, encoder = prepare_dataset("data/fluoride_data.csv", verbose=False)
    
    # Get model from utils
    try:
        model = get_model(internal_name)
    except:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    # Check if it needs classification target (XGBoost logic in utils might be classifier)
    # The current utils has RandomForestRegressor for 'rf_regressor'
    # For others like XGBoost, it returns XGBClassifier. We need regressor for Fluoride values.
    # I'll force regressor types where possible.
    if "regressor" not in internal_name and internal_name not in ["svr", "linearregression"]:
        # Fallback to RF Regressor if utils only has classifiers for those names
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X, df_orig[target_col])
    
    # Save artifacts to ensure persistence
    try:
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        if encoder:
            joblib.dump(encoder, encoder_path)
    except:
        pass
    
    artifacts = {
        'model': model,
        'scaler': scaler,
        'encoder': encoder,
        'features': features,
        'df_orig': df_orig
    }
    return artifacts

def haversine(lat1, lon1, lat2, lon2):
    import math
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2.0)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2.0)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def state_centroids():
    return {
        'ANDHRA PRADESH': (15.9129, 79.7400),
        'ASSAM': (26.2006, 92.9376),
        'BENGAL': (23.6850, 90.3563),
        'BIHAR': (25.0961, 85.3131),
        'GOA': (15.3800, 73.8170),
        'GUJARAT': (22.2587, 71.1924),
        'HARYANA': (29.0588, 76.0856),
        'KARNATAKA': (15.3173, 75.7139),
        'KERALA': (10.8505, 76.2711),
        'MADHYA PRADESH': (22.9734, 78.6569),
        'MAHARASHTRA': (19.7515, 75.7139),
        'ORISSA': (20.9517, 85.0985),
        'PUNJAB': (31.1471, 75.3412),
        'RAJASTHAN': (27.0238, 74.2179),
        'TAMIL NADU': (11.1271, 78.6569),
        'TELANGANA': (18.1124, 79.0193),
        'UTTAR PRADESH': (26.8467, 80.9462),
        'WEST BENGAL': (22.9868, 87.8550),
        'DELHI': (28.7041, 77.1025),
    }

def get_location_coords(query, df):
    if not query:
        return None, None
        
    query = str(query).strip().upper()
    
    centroids = state_centroids()
    for state, coords in centroids.items():
        if state in query or query in state:
            if len(query) > 3: 
                return coords

    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    
    if not (lat_col and lon_col):
        return None, None

    search_cols = [c for c in df.columns if any(x in c.lower() for x in ['pin', 'dist', 'state', 'loc'])]
    
    for col in search_cols:
        matches = df[df[col].astype(str).str.upper().str.contains(query, na=False)]
        if not matches.empty:
            return matches[lat_col].mean(), matches[lon_col].mean()
            
    return None, None

# -----------------------------------------------------------------------------
# Stage 1: Landing Page
# -----------------------------------------------------------------------------
def render_landing_page():
    st.markdown(f"<div style='text-align: center; margin-top: 50px;'><h1>AquaRisk Intelligence</h1><p style='font-size: 1.2rem; color: #555;'>Advanced/Sustainability/Safety</p></div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### USER ACCESS")
        with st.container(border=True):
            name = st.text_input("FULL NAME")
            place = st.text_input("DISTRICT OR PINCODE")
            
            if name and place:
                st.session_state.user_name = name
                st.session_state.user_place = place
                
                # Logic to find nearest wells (Triggered by button)
                if st.button("FIND NEAREST SAFE LOCATIONS"):
                    df, target_col = load_data()
                    lat, lon = get_location_coords(place, df)
                    
                    if lat and lon:
                        st.success(f"Location Identified: {lat:.4f}, {lon:.4f}")
                        
                        st.markdown("### LOCAL SAFETY ALERT")
                        
                        # Find nearest
                        df['dist'] = df.apply(lambda row: haversine(lat, lon, row.get('Latitude', 0), row.get('Longitude', 0)), axis=1)
                        
                        # Columns to show
                        loc_col = next((c for c in df.columns if 'location' in c.lower()), 'Locations')
                        t_col = target_col if target_col and target_col in df.columns else df.columns[-1]
                        
                        nearest = df.nsmallest(5, 'dist')[[loc_col, 'dist', t_col]]
                        
                        # Style the table
                        def get_risk(f):
                            try:
                                f = float(f)
                            except:
                                return "UNKNOWN", "gray"
                            if f < 1.0: return "SAFE", "green"
                            elif f < 1.5: return "CAUTION", "orange"
                            return "DANGEROUS", "red"
                            
                        nearest['Risk'], nearest['Color'] = zip(*nearest[t_col].map(get_risk))
                        
                        # Custom HTML Table
                        table_html = "<table style='width:100%; border-collapse: collapse; background-color: #252526; color: #E0E0E0;'>"
                        table_html += f"<tr style='background-color: #333; text-align: left; border-bottom: 2px solid #555; color: {COLOR_TEAL};'><th>LOCATION</th><th>DISTANCE (KM)</th><th>{(t_col or 'Value').upper()}</th><th>RISK LEVEL</th></tr>"
                        for _, row in nearest.iterrows():
                            color = row['Color']
                            val_disp = f"{row[t_col]:.2f}" if isinstance(row[t_col], (int, float)) else str(row[t_col])
                            table_html += f"<tr style='border-bottom: 1px solid #444; color: #E0E0E0;'><td>{row[loc_col]}</td><td>{row['dist']:.2f}</td><td>{val_disp}</td><td style='color:{color}; font-weight:bold;'>{row['Risk']}</td></tr>"
                        table_html += "</table>"
                        
                        st.markdown(table_html, unsafe_allow_html=True)
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                    else:
                        if len(place) > 0:
                            st.warning("Could not geocode location exactly. Showing general access.")

                st.markdown("<br>", unsafe_allow_html=True)

                if st.button("PROCEED TO TECHNICAL DASHBOARD"):
                    st.session_state.stage = 'dashboard'
                    st.rerun()

# -----------------------------------------------------------------------------
# Stage 2: Dashboard
# -----------------------------------------------------------------------------
def render_dashboard():
    # Sidebar
    st.sidebar.markdown(f"## WELCOME, {st.session_state.get('user_name', 'USER')}")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("App Settings")
    selected_model = st.sidebar.selectbox(
        "Prediction Model", 
        ["Random Forest", "XGBoost", "Linear Regression", "SVR"],
        index=0
    )
    
    if st.sidebar.button("LOGOUT"):
        st.session_state.stage = 'landing'
        st.rerun()
        
    st.markdown("## TECHNICAL SUSTAINABILITY DASHBOARD")
    
    # Load resources with selected model
    artifacts = get_app_resources(selected_model)
    df, target_col = load_data()
    
    # Columns: Input (1), Map (2), Risk/Action (1)
    c_left, c_mid, c_right = st.columns([1, 2, 1])
    
    # --- Left: INPUTS ---
    with c_left:
        st.markdown("### WATER PARAMETERS")
        st.caption("Adjust sliders to simulate water sample values.")
        with st.container(border=True):
            # Reactive Sliders (Streamlit reacts on change by default)
            ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1)
            ec = st.slider("EC (µS/cm)", 0.0, 3000.0, 500.0, 10.0)
            na = st.slider("Sodium (Na+)", 0.0, 1000.0, 50.0, 1.0)
            mg = st.slider("Magnesium (Mg2+)", 0.0, 500.0, 30.0, 1.0)
            ca = st.slider("Calcium (Ca2+)", 0.0, 500.0, 40.0, 1.0)
            hco3 = st.slider("Bicarbonate (HCO3-)", 0.0, 1000.0, 200.0, 10.0)
            
            user_inputs = {'pH': ph, 'EC': ec, 'Na': na, 'Mg': mg, 'Ca': ca, 'HCO3': hco3}
    
    # --- Prediction Logic (Live Risk Analysis) ---
    model = artifacts['model']
    features = artifacts['features']
    
    # Construct input array
    input_data = {f: 0.0 for f in features}
    for k, v in user_inputs.items():
        for f in features:
            if k.lower() in f.lower():
                input_data[f] = v
                
    input_df = pd.DataFrame([input_data])
    
    # Scale inputs (CRITICAL: Model trained on scaled data)
    scaler = artifacts.get('scaler')
    if scaler:
        try:
           # scaler expects specific columns. 
           # If features == numeric_features, we can just transform.
           # But to be safe, we should ensure column order matches scaler.feature_names_in_
           if hasattr(scaler, 'feature_names_in_'):
               input_df = input_df[scaler.feature_names_in_]
           input_df = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        except Exception as e:
           st.warning(f"Scaling failed: {e}")

    # Predict
    try:
        pred_val = model.predict(input_df)[0]
    except Exception as e:
        st.error(f"Prediction Error: {e}")
        st.write(f"Model expects: {len(features)} features")
        st.write(f"Input provided: {input_df.shape[1]} features")
        # st.write("Expected:", features)
        # st.write("Provided:", input_df.columns.tolist())
        pred_val = 1.2  
        
    label, risk_score = fuzzy_classify_fluoride(pred_val)
    
    # --- Center: MAP (Dynamic Mapping) ---
    with c_mid:
        st.markdown("### REGIONAL TRENDS")
        # Center map on India or user location
        # Default start coords
        start_coords = [20.5937, 78.9629] # India center
        zoom_start = 5

        # Try to center on user location
        user_place = st.session_state.get('user_place', '')
        if user_place:
            try:
                lat, lon = get_location_coords(user_place, df)
                if lat and lon:
                    start_coords = [lat, lon]
                    zoom_start = 9
            except:
                pass
        
        m = folium.Map(location=start_coords, zoom_start=zoom_start, tiles='CartoDB dark_matter') # Dark map tiles
        
        # Prepare Heatmap Data
        lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
        lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
        f_col = target_col if target_col else next((c for c in df.columns if 'fluoride' in c.lower() or 'f_raw' in c.lower() or c=='F'), None)
        loc_col = next((c for c in df.columns if 'location' in c.lower()), 'Locations')
        
        # Debug info for empty map issues
        if not (lat_col and lon_col and f_col):
            st.error(f"Map Data Missing Columns: Lat={lat_col}, Lon={lon_col}, Target={f_col}")
            st.write("Available Columns:", df.columns.tolist())
        
        if lat_col and lon_col and f_col:
            df_clean = df.dropna(subset=[lat_col, lon_col, f_col])
            
            # High-Density Heatmap Gradient (Olive, Goldenrod, Burnt Red)
            heat_data = df_clean[[lat_col, lon_col, f_col]].values.tolist()
            gradient = {0.4: '#A5A178', 0.65: '#FFC857', 1.0: '#DB3A34'} 
            plugins.HeatMap(heat_data, radius=15, blur=10, min_opacity=0.4, gradient=gradient).add_to(m)
            
            # On-Click Popups (using CircleMarkers)
            for idx, row in df_clean.iterrows():
                try:
                    val = float(row[f_col])
                    risk_txt = "High Risk" if val > 1.5 else "Safe" if val < 1.0 else "Medium Risk"
                    popup_html = f"""
                    <b>{row.get(loc_col, 'Unknown Location')}</b><br>
                    Mean Fluoride: {val:.2f} mg/L<br>
                    Calculated Risk: {risk_txt}<br>
                    Recommended Action: {'Filter Water' if val > 1.0 else 'Safe to Consume'}
                    """
                    folium.CircleMarker(
                        location=[row[lat_col], row[lon_col]],
                        radius=3,
                        color='transparent',
                        fill=True,
                        fill_color='transparent',
                        fill_opacity=0.0,
                        popup=folium.Popup(popup_html, max_width=250)
                    ).add_to(m)
                except:
                    pass
            
        st_folium(m, width=None, height=500, use_container_width=True)
        
    # --- Right: GAUGE & ACTION ---
    with c_right:
        st.markdown("### RISK ANALYSIS")
        
        # Gauge Chart
        st.caption("Real-time prediction based on your inputs.")
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred_val,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Predicted Fluoride (mg/L)"},
            gauge = {
                'axis': {'range': [None, 5], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': COLOR_TEAL},
                'bgcolor': "#252526",
                'borderwidth': 2,
                'bordercolor': "#555",
                'steps': [
                    {'range': [0, 1.5], 'color': "#A5A178"}, # Olive Green (Safe)
                    {'range': [1.5, 10], 'color': "#DB3A34"}], # Burnt Red (High Risk)
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 1.5}
            }
        ))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Status:** {label}")
        st.markdown(f"**Fuzzy Score:** {risk_score:.2f}")
        
        st.markdown("---")
        st.markdown("### ACTIONABLES")
        st.markdown("Links to NGOs & Support:")
        st.markdown("- [WaterAid India](https://www.wateraid.org/in/)")
        st.markdown("- [Sankalpa Rural Development](https://sankalparural.org/)")
        st.markdown("- [Fluoride Knowledge Action Network](http://www.fluorideindia.org/)")
        
        st.info("Ensure to test water samples at a certified lab before consumption.")

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
def main():
    inject_custom_css()
    
    if 'stage' not in st.session_state:
        st.session_state.stage = 'landing'
        
    if st.session_state.stage == 'landing':
        render_landing_page()
    else:
        render_dashboard()

if __name__ == "__main__":
    main()
