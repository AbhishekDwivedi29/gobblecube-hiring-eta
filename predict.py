"""Submission interface — this is what Gobblecube's grader imports."""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

# 1. Update the filename to load the new multi-tier model
_MODEL_PATH = Path(__file__).parent / "hourly_lookup_model.pkl"

# Load the dictionary at module import time (happens once)
with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)

# 2. Unpack all three tiers of our lookup table
_HOURLY_ZONE_MEDIANS = _MODEL["hourly_zone_medians"]
_ZONE_MEDIANS = _MODEL["zone_medians"]
_GLOBAL_MEDIAN = _MODEL["global_median"]

def predict(request: dict) -> float:
    """Predict trip duration in seconds using a time-aware dictionary lookup.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime (e.g., "2023-01-01T15:30:00")
            "passenger_count": int,
        }
    """
    pickup = int(request["pickup_zone"])
    dropoff = int(request["dropoff_zone"])
    
    # Extract the hour from the ISO 8601 string
    # E.g., "2023-01-01T15:30:00" -> 15
    dt = datetime.fromisoformat(request["requested_at"])
    hour = dt.hour

    # Tier 1: Attempt to look up the exact route at the exact hour
    exact_match = _HOURLY_ZONE_MEDIANS.get((pickup, dropoff, hour))
    if exact_match is not None:
        return float(exact_match)

    # Tier 2: Fall back to the route average across all times
    route_match = _ZONE_MEDIANS.get((pickup, dropoff))
    if route_match is not None:
        return float(route_match)

    # Tier 3: Complete fallback to the global median
    return float(_GLOBAL_MEDIAN)