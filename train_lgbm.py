#!/usr/bin/env python
"""
Phase 3: LightGBM Baseline

Trains a LightGBM model optimizing directly for MAE.
Handles categorical features natively without one-hot encoding.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "lgbm_model.pkl"

# We treat almost everything as a category!
CATEGORICAL_FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "month"]
NUMERIC_FEATURES = ["passenger_count"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract features and cast categoricals to the correct dtype."""
    ts = pd.to_datetime(df["requested_at"])
    
    df_features = pd.DataFrame({
        "pickup_zone":     df["pickup_zone"],
        "dropoff_zone":    df["dropoff_zone"],
        "hour":            ts.dt.hour,
        "dow":             ts.dt.dayofweek,
        "month":           ts.dt.month,
        "passenger_count": df["passenger_count"].astype("int8"),
    })
    
    # LightGBM requires categorical columns to literally be pandas 'category' dtype
    for col in CATEGORICAL_FEATURES:
        df_features[col] = df_features[col].astype("category")
        
    return df_features[FEATURES]

def main() -> None:
    print("Loading data...")
    train = pd.read_parquet(DATA_DIR / "train.parquet")
    dev = pd.read_parquet(DATA_DIR / "dev.parquet")

    print("Engineering features...")
    X_train = engineer_features(train)
    y_train = train["duration_seconds"].to_numpy()
    
    X_dev = engineer_features(dev)
    y_dev = dev["duration_seconds"].to_numpy()

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES)
    dev_data = lgb.Dataset(X_dev, label=y_dev, reference=train_data, categorical_feature=CATEGORICAL_FEATURES)

    print("\nTraining LightGBM directly on MAE (L1 Loss)...")
    params = {
        "objective": "mae",       
        "metric": "mae",
        "learning_rate": 0.1,
        "num_leaves": 63,         # Slightly deeper trees to capture zone-pair interactions
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

    # Save the model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved LightGBM model to {MODEL_PATH}")

if __name__ == "__main__":
    main()