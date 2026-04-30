#!/usr/bin/env python
"""
Phase 4: LightGBM Baseline with Polars

Trains a LightGBM model optimizing directly for MAE.
Uses Polars for ultra-fast, RAM-safe feature engineering and coordinate merging.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "lgbm_model.pkl"

CATEGORICAL_FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "month", "is_weekend"]
NUMERIC_FEATURES = ["passenger_count", "distance_km"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Load coordinates lazily
ZONE_COORDS = pl.scan_csv(DATA_DIR / "zone_coords.csv")

def haversine_expr(lat1: str, lon1: str, lat2: str, lon2: str) -> pl.Expr:
    """Creates a Polars Expression to calculate Haversine distance natively."""
    # Convert degrees to radians
    rad_lat1 = pl.col(lat1) * (np.pi / 180)
    rad_lon1 = pl.col(lon1) * (np.pi / 180)
    rad_lat2 = pl.col(lat2) * (np.pi / 180)
    rad_lon2 = pl.col(lon2) * (np.pi / 180)
    
    dlat = rad_lat2 - rad_lat1
    dlon = rad_lon2 - rad_lon1
    
    a = (dlat / 2).sin()**2 + rad_lat1.cos() * rad_lat2.cos() * (dlon / 2).sin()**2
    distance = 6371.0 * 2 * a.sqrt().arcsin()
    
    # Return as a 32-bit float to save RAM
    return distance.cast(pl.Float32).alias("distance_km")

def engineer_features_polars(parquet_path: Path) -> pd.DataFrame:
    """Uses Polars lazy evaluation to process data, returning a lightweight Pandas DF for LightGBM."""
    
    # 1. SCAN the file (does not load into RAM yet!)
    df = pl.scan_parquet(parquet_path)
    
    # 2. Merge Pickup Coordinates
    df = df.join(ZONE_COORDS, left_on="pickup_zone", right_on="LocationID", how="left")
    df = df.rename({"lat": "pickup_lat", "lon": "pickup_lon"})
    
    # 3. Merge Dropoff Coordinates
    df = df.join(ZONE_COORDS, left_on="dropoff_zone", right_on="LocationID", how="left")
    df = df.rename({"lat": "dropoff_lat", "lon": "dropoff_lon"})
    
    # 4. Generate all features in parallel
    df = df.with_columns(
        # First, convert the string column to an actual datetime object
        pl.col("requested_at").str.to_datetime()
        ).with_columns([
        pl.col("requested_at").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("requested_at").dt.weekday().cast(pl.Int8).alias("dow"), # 1=Mon, ..., 7=Sun
        pl.col("requested_at").dt.month().cast(pl.Int8).alias("month"),
        pl.col("requested_at").dt.weekday().is_in([6, 7]).cast(pl.Int8).alias("is_weekend"),
        pl.col("passenger_count").cast(pl.Int8),
        haversine_expr("pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon")
    ])

    
    # 5. Drop everything we don't need, and finally COLLECT (execute the pipeline)
    cols_to_keep = FEATURES + ["duration_seconds"]
    df = df.select(cols_to_keep).collect()
    
    # 6. LightGBM expects pandas DataFrames to natively detect categories
    df_pd = df.to_pandas()
    for col in CATEGORICAL_FEATURES:
        df_pd[col] = df_pd[col].astype("category")
        
    return df_pd

def main() -> None:
    print("Engineering training data with Polars...")
    train = engineer_features_polars(DATA_DIR / "train.parquet")
    X_train = train[FEATURES]
    y_train = train["duration_seconds"].to_numpy()
    
    print("Engineering dev data with Polars...")
    dev = engineer_features_polars(DATA_DIR / "dev.parquet")
    X_dev = dev[FEATURES]
    y_dev = dev["duration_seconds"].to_numpy()

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    dev_data = lgb.Dataset(X_dev, label=y_dev, reference=train_data, categorical_feature=CATEGORICAL_FEATURES)

    print("\nTraining LightGBM directly on MAE (L1 Loss)...")
    params = {
        "objective": "mae",       
        "metric": "mae",
        "learning_rate": 0.1,
        "num_leaves": 63,        
        "max_depth": -1,
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1
    }

    t0 = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, dev_data],
        valid_names=["train", "dev"],
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(50)]
    )
    print(f"  Trained in {time.time() - t0:.0f}s")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved LightGBM model to {MODEL_PATH}")

if __name__ == "__main__":
    main()