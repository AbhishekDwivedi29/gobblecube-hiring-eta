"""Submission interface — this is what Gobblecube's grader imports."""

from __future__ import annotations

import pickle
from pathlib import Path

_MODEL_PATH = Path(__file__).parent / "lookup_model.pkl"

# Load the dictionary at module import time (happens once)
with open(_MODEL_PATH, "rb") as _f:
    _MODEL = pickle.load(_f)

_ZONE_MEDIANS = _MODEL["zone_medians"]
_GLOBAL_MEDIAN = _MODEL["global_median"]

def predict(request: dict) -> float:
    """Predict trip duration in seconds using a simple dictionary lookup.

    Input schema:
        {
            "pickup_zone":     int,   # NYC taxi zone, 1-265
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601 datetime
            "passenger_count": int,
        }
    """
    pickup = int(request["pickup_zone"])
    dropoff = int(request["dropoff_zone"])

    # Attempt to look up the specific zone pair.
    # If the pair was never seen in training data, fall back to the global median.
    return float(_ZONE_MEDIANS.get((pickup, dropoff), _GLOBAL_MEDIAN))