"""Submission interface for LightGBM."""

from __future__ import annotations

import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd

_DATA_DIR = Path(__file__).parent / "data"
_MODEL_PATH = Path(__file__).parent / "lgbm_model.pkl"

# 1. Load the model
with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)

# 2. Load the zone coordinates for fast dictionary-like lookup
_ZONE_COORDS = pd.read_csv(_DATA_DIR / "zone_coords.csv").set_index("LocationID")

CATEGORICAL_FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "month", "is_weekend"]

def calculate_haversine(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two points on the fly."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))

def predict(request: dict) -> float:
    ts = datetime.fromisoformat(request["requested_at"])
    
    pickup = int(request["pickup_zone"])
    dropoff = int(request["dropoff_zone"])
    
    # 3. Look up coordinates for the requested zones
    # Using .get() with a default of 0 in case a zone is missing
    pick_lat = _ZONE_COORDS.loc[pickup, "lat"] if pickup in _ZONE_COORDS.index else 0
    pick_lon = _ZONE_COORDS.loc[pickup, "lon"] if pickup in _ZONE_COORDS.index else 0
    drop_lat = _ZONE_COORDS.loc[dropoff, "lat"] if dropoff in _ZONE_COORDS.index else 0
    drop_lon = _ZONE_COORDS.loc[dropoff, "lon"] if dropoff in _ZONE_COORDS.index else 0
    
    distance_km = calculate_haversine(pick_lat, pick_lon, drop_lat, drop_lon)
    
    # 4. FIX: Polars weekday is 1-7, Python datetime is 0-6. Add 1!
    polars_dow = ts.weekday() + 1
    
    # Create a 1-row dataframe containing ALL features the model trained on
    df = pd.DataFrame({
        "pickup_zone":     [pickup],
        "dropoff_zone":    [dropoff],
        "hour":            [ts.hour],
        "dow":             [polars_dow],
        "month":           [ts.month],
        "is_weekend":      [1 if polars_dow in [6, 7] else 0],
        "passenger_count": [int(request["passenger_count"])],
        "distance_km":     [distance_km]  # Added the new feature
    })
    
    # Apply category dtypes so LightGBM recognizes them natively
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
        
    # LightGBM predict returns an array, grab the first element
    return float(_MODEL.predict(df)[0])