#!/usr/bin/env python
"""
Phase 5: LightGBM + Polars + Optuna (Bayesian Optimization)

Uses Bayesian Optimization to automatically find the perfect balance
between underfitting and overfitting to minimize MAE.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import polars as pl
import pandas as pd
import lightgbm as lgb
import optuna  # <-- The Bayesian Optimizer

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "lgbm_optuna_best_model.pkl"

CATEGORICAL_FEATURES = ["pickup_zone", "dropoff_zone", "hour", "dow", "month", "is_weekend"]
NUMERIC_FEATURES = ["distance_km"] 
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Load coordinates lazily
ZONE_COORDS = pl.scan_csv(DATA_DIR / "zone_coords.csv")

def haversine_expr(lat1: str, lon1: str, lat2: str, lon2: str) -> pl.Expr:
    rad_lat1 = pl.col(lat1) * (np.pi / 180)
    rad_lon1 = pl.col(lon1) * (np.pi / 180)
    rad_lat2 = pl.col(lat2) * (np.pi / 180)
    rad_lon2 = pl.col(lon2) * (np.pi / 180)
    
    dlat = rad_lat2 - rad_lat1
    dlon = rad_lon2 - rad_lon1
    
    a = (dlat / 2).sin()**2 + rad_lat1.cos() * rad_lat2.cos() * (dlon / 2).sin()**2
    distance = 6371.0 * 2 * a.sqrt().arcsin()
    return distance.cast(pl.Float32).alias("distance_km")

def engineer_features_polars(parquet_path: Path) -> pd.DataFrame:
    df = pl.scan_parquet(parquet_path)
    df = df.join(ZONE_COORDS, left_on="pickup_zone", right_on="LocationID", how="left")
    df = df.rename({"lat": "pickup_lat", "lon": "pickup_lon"})
    df = df.join(ZONE_COORDS, left_on="dropoff_zone", right_on="LocationID", how="left")
    df = df.rename({"lat": "dropoff_lat", "lon": "dropoff_lon"})
    
    df = df.with_columns(
        pl.col("requested_at").str.to_datetime()
    ).with_columns([
        pl.col("requested_at").dt.hour().cast(pl.Int8).alias("hour"),
        pl.col("requested_at").dt.weekday().cast(pl.Int8).alias("dow"),
        pl.col("requested_at").dt.month().cast(pl.Int8).alias("month"),
        # FIXED: Polars uses ISO standard where 6=Sat, 7=Sun
        pl.col("requested_at").dt.weekday().is_in([6, 7]).cast(pl.Int8).alias("is_weekend"),
        haversine_expr("pickup_lat", "pickup_lon", "dropoff_lat", "dropoff_lon")
    ])
    
    cols_to_keep = FEATURES + ["duration_seconds"]
    df = df.select(cols_to_keep).collect()
    
    df_pd = df.to_pandas()
    for col in CATEGORICAL_FEATURES:
        df_pd[col] = df_pd[col].astype("category")
        
    return df_pd

def objective(trial):
    """The Optuna Bayesian Optimization Function"""
    
    print(f"\n--- [Optuna] Starting Trial {trial.number} ---")
    
    # 1. Define the search space
    params = {
        "objective": "mae",
        "metric": "mae",
        "verbosity": -1, 
        "n_jobs": -1,
        "random_state": 42,
        
        # Bayesian Search Space (Optuna guesses these)
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "cat_l2": trial.suggest_float("cat_l2", 1.0, 50.0)
    }
    
    # 2. Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[dev_data],
        valid_names=["dev"], # Explicitly name the validation set
        callbacks=[
            lgb.early_stopping(stopping_rounds=15, verbose=False),
            lgb.log_evaluation(50)  # Print every 50 rounds so you know it's working
        ]
    )

    # 3. Return the best validation score
    best_score = model.best_score["dev"]["l1"]
    print(f"--- Trial {trial.number} finished with MAE: {best_score:.4f} ---\n")
    return best_score

if __name__ == "__main__":
    print("1/4: Engineering training data with Polars...")
    train = engineer_features_polars(DATA_DIR / "train.parquet")
    X_train = train[FEATURES]
    y_train = train["duration_seconds"].to_numpy()
    
    print("2/4: Engineering dev data with Polars...")
    dev = engineer_features_polars(DATA_DIR / "dev.parquet")
    X_dev = dev[FEATURES]
    y_dev = dev["duration_seconds"].to_numpy()

    print("3/4: Loading data into LightGBM Engine (This takes a moment)...")
    global train_data, dev_data
    
    # CRITICAL FIX 1: max_bin goes here, not in the objective function
    dataset_params = {"max_bin": 127}
    
    # CRITICAL FIX 2: free_raw_data=False. This stops LightGBM from deleting your RAM 
    # after Trial 0, which allows Optuna to reuse the data for all 10 trials.
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=CATEGORICAL_FEATURES, params=dataset_params, free_raw_data=False)
    dev_data = lgb.Dataset(X_dev, label=y_dev, reference=train_data, categorical_feature=CATEGORICAL_FEATURES, free_raw_data=False)

    print("\n4/4: Starting Bayesian Optimization with Optuna...")
    study = optuna.create_study(direction="minimize")
    
    # Run 10 experiments
    study.optimize(objective, n_trials=10, show_progress_bar=True)

    print("\n==================================")
    print(f"WINNING SCORE (MAE): {study.best_value:.4f}")
    print("WINNING PARAMETERS:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("==================================\n")

    print("Retraining final production model on Winning Parameters...")
    final_params = study.best_params
    final_params.update({"objective": "mae", "metric": "mae", "n_jobs": -1, "verbose": -1, "random_state": 42})
    
    final_model = lgb.train(
        final_params,
        train_data,
        num_boost_round=1500,
        valid_sets=[train_data, dev_data],
        valid_names=["train", "dev"],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=True), lgb.log_evaluation(50)]
    )

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\nSUCCESS! Saved Best LightGBM model to {MODEL_PATH}")