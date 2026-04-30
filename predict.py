"""Submission interface for LightGBM."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd

_MODEL_PATH = Path(__file__).parent / "lgbm_model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)
CATEGORICAL_FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "month", "is_weekend"]

def predict(request: dict) -> float:
    ts = datetime.fromisoformat(request["requested_at"])
    
    # Create a 1-row dataframe
    df = pd.DataFrame({
    "pickup_zone":     [int(request["pickup_zone"])],
    "dropoff_zone":    [int(request["dropoff_zone"])],
    "hour":            [ts.hour],
    "dow":             [ts.weekday()],
    "month":           [ts.month],
    "is_weekend":      [1 if ts.weekday() in [5, 6] else 0],  # ADD THIS
    "passenger_count": [int(request["passenger_count"])],
})
    
    # Apply category dtypes so LightGBM recognizes them
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
        
    # LightGBM predict returns an array, grab the first element
    return float(_MODEL.predict(df)[0])