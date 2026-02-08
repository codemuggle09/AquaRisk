"""
Preprocessing for Fluoride dataset (dataset1.csv).
Auto-detects chemical features, cleans data, normalizes, and creates fluoride risk classes.
Categorical features are one-hot encoded automatically.
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define chemical features
# Define chemical features
FEATURE_KEYWORDS = ['pH', 'TDS', 'EC', 'Na', 'Mg', 'Ca', 'K', 'HCO3', 'SO4', 'ClO4', 'Cl', 'NO3']
TARGET_KEYWORDS = ['Fluoride', 'F-'] # Removed single 'F' to avoid matching 'Coliform'
EXCLUDED_KEYWORDS = ['state', 'district', 'location', 'name', 'year', 'date', 'block', 'village', 'hab', 'coliform']

def normalize_col(col):
    """Remove units and extra characters from column name"""
    col = re.sub(r'\(.*?\)', '', str(col))
    return col.strip().lower()

def detect_columns(df):
    """Auto-detect target and feature columns"""
    cols = {normalize_col(c): c for c in df.columns}
    target = None
    for key in TARGET_KEYWORDS:
        for k, v in cols.items():
            if key.lower() in k:
                target = v
                break
        if target:
            break

    features = []
    for key in FEATURE_KEYWORDS:
        for k, v in cols.items():
            # Check exclusion list
            if any(ex in k for ex in EXCLUDED_KEYWORDS):
                continue
                
            if key.lower() in k and v != target:
                features.append(v)
                break
    return target, features

def state_centroids():
    return {
        'ANDHRA PRADESH': (15.9129, 79.7400), 'ASSAM': (26.2006, 92.9376),
        'BENGAL': (23.6850, 90.3563), 'BIHAR': (25.0961, 85.3131),
        'GOA': (15.3800, 73.8170), 'GUJARAT': (22.2587, 71.1924),
        'HARYANA': (29.0588, 76.0856), 'KARNATAKA': (15.3173, 75.7139),
        'KERALA': (10.8505, 76.2711), 'MADHYA PRADESH': (22.9734, 78.6569),
        'MAHARASHTRA': (19.7515, 75.7139), 'ORISSA': (20.9517, 85.0985),
        'PUNJAB': (31.1471, 75.3412), 'RAJASTHAN': (27.0238, 74.2179),
        'TAMIL NADU': (11.1271, 78.6569), 'TELANGANA': (18.1124, 79.0193),
        'UTTAR PRADESH': (26.8467, 80.9462), 'WEST BENGAL': (22.9868, 87.8550),
        'DELHI': (28.7041, 77.1025), 'ODISHA': (20.9517, 85.0985)
    }

def prepare_dataset(path, verbose=True):
    """Load, clean, normalize dataset and create fluoride class labels"""
    df = pd.read_csv(path, encoding='latin1')
    target_col, features = detect_columns(df)
    if verbose:
        print(f"Detected Target: {target_col}")
        print(f"Detected Features: {features}")

    # Replace common non-numeric placeholders with NaN
    df.replace(['-', '—', 'NA', 'na', ''], np.nan, inplace=True)

    # Detect numeric vs categorical features
    numeric_features = []
    categorical_features = []
    for col in features:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].notna().any():
                numeric_features.append(col)
            else:
                categorical_features.append(col)
        except:
            categorical_features.append(col)

    # Impute numeric features
    imputer = SimpleImputer(strategy='median')
    cols_to_impute = numeric_features
    if target_col:
        cols_to_impute = numeric_features + [target_col]
        
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    # Store raw target for classification
    # If target column is missing, synthesize it for demo purposes
    if not target_col:
        print("Target 'Fluoride' detected missing. Synthesizing data based on parameters...")
        # Logic: F = 0.5 + 0.001*EC - 0.05*pH + noise (purely illustrative)
        # Using numeric features available
        ph_col = next((c for c in features if 'pH' in c), None)
        ec_col = next((c for c in features if 'EC' in c or 'CONDUCTIVITY' in c), None)
        
        base_val = 1.0 # Base safe level
        if ph_col:
            df[ph_col] = pd.to_numeric(df[ph_col], errors='coerce').fillna(7.0)
            base_val += (df[ph_col] - 7.0) * 0.2
        if ec_col:
            df[ec_col] = pd.to_numeric(df[ec_col], errors='coerce').fillna(500.0)
            base_val += (df[ec_col] / 1000.0) * 0.5
            
        noise = np.random.normal(0, 0.3, len(df))
        df['F_raw'] = np.clip(base_val + noise, 0.1, 5.0)
        target_col = 'Synthesized Fluoride'
        df[target_col] = df['F_raw']
    else:
        df['F_raw'] = df[target_col]

    # If Lat/Lon missing, synthesize based on State
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    
    if not lat_col or not lon_col:
        print("Coordinates missing. Synthesizing based on State centroids...")
        centroids = state_centroids()
        state_col = next((c for c in df.columns if 'state' in c.lower()), None)
        
        if state_col:
            def get_synth_coords(row):
                s = str(row[state_col]).strip().upper()
                c = centroids.get(s, (20.5937, 78.9629)) # Default India
                # Add huge jitter to spread across state
                return c[0] + np.random.uniform(-1.5, 1.5), c[1] + np.random.uniform(-1.5, 1.5)
            
            df['Latitude'], df['Longitude'] = zip(*df.apply(get_synth_coords, axis=1))
        else:
             # Just random India points
             df['Latitude'] = np.random.uniform(8.0, 37.0, len(df))
             df['Longitude'] = np.random.uniform(68.0, 97.0, len(df))

    # Re-detect numeric features since we might have synthesized some
    # Or just proceed with what we have.
    # The crucial part above sets 'F_raw' which drives the labels.

    # Risk classification (USEPA)
    df['fluoride_class'] = pd.cut(
        df['F_raw'], bins=[-1, 1.5, 2.5, df['F_raw'].max() + 1], labels=[0, 1, 2]
    ).astype(int)

    # Normalize numeric features
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])

    # One-hot encode categorical features
    encoder = None
    if categorical_features:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(df_scaled[categorical_features])
        X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_features), index=df_scaled.index)
        X = pd.concat([df_scaled[numeric_features], X_cat_df], axis=1)
    else:
        X = df_scaled[numeric_features]

    y = df['fluoride_class'].astype(int)


    if verbose:
        print(f"Dataset Shape: {df.shape}")
        print(f"Numeric Features: {numeric_features}")
        print(f"Categorical Features: {categorical_features}")
        print(f"Number of NaNs after preprocessing: {X.isna().sum().sum()}")

    return X, y, df_scaled[target_col], df, features, target_col, scaler, encoder


