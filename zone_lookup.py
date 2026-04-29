#!/usr/bin/env python
"""
True Baseline: Zone-Pair Median Lookup.

This script groups the training data by (pickup_zone, dropoff_zone)
and calculates the median trip duration. It saves these medians, along
with a global median fallback, into a pickle file for inference.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "lookup_model.pkl"

def main() -> None:
    train_path = DATA_DIR / "train.parquet"
    dev_path = DATA_DIR / "dev.parquet"

    if not train_path.exists() or not dev_path.exists():
        raise SystemExit("Missing data files. Run `python data/download_data.py` first.")

    print("Loading data...")
    train = pd.read_parquet(train_path)
    dev = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,} rows")
    print(f"  dev:   {len(dev):,} rows")

    print("\nCalculating medians...")
    t0 = time.time()

    # 1. Calculate the global median as a fallback for unseen pairs
    global_median = float(train["duration_seconds"].median())

    # 2. Calculate the median for each pickup/dropoff pair
    # We use median instead of mean because it is robust to extreme outliers (like a 5-hour traffic jam)
    zone_medians = (
        train.groupby(["pickup_zone", "dropoff_zone"])["duration_seconds"]
        .median()
        .to_dict()
    )

    print(f"  computed {len(zone_medians):,} unique zone pairs in {time.time() - t0:.2f}s")

    # Create our "model" artifact
    model_artifact = {
        "global_median": global_median,
        "zone_medians": zone_medians
    }

    # Save the lookup dictionary
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_artifact, f)
    print(f"Saved lookup model to {MODEL_PATH}")

    # Quick Local Evaluation on Dev
    print("\nEvaluating on Dev...")
    def predict_row(row):
        return zone_medians.get((row.pickup_zone, row.dropoff_zone), global_median)

    preds = dev.apply(predict_row, axis=1).to_numpy()
    y_dev = dev["duration_seconds"].to_numpy()

    mae = float(np.mean(np.abs(preds - y_dev)))
    print(f"Dev MAE (Lookup Baseline): {mae:.1f} seconds")


if __name__ == "__main__":
    main()