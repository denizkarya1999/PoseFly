"""Shared confidence defaults and validation, independent of the ML runtime."""

import math

MODEL_LABELS = {
    "drone": "Drone",
    "angle": "Angle",
    "distance": "Distance",
    "led": "LED ID",
    "speed": "Speed",
}
DEFAULT_CONFIDENCE = {"drone": 0.25, "angle": 0.001, "distance": 0.001, "led": 0.001, "speed": 0.001}


def confidence_thresholds(values=None):
    """Resolve partial settings; reject invalid values before invoking models."""
    thresholds = DEFAULT_CONFIDENCE.copy()
    if values is None:
        return thresholds
    for name in thresholds:
        value = float(values.get(name, thresholds[name]))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{MODEL_LABELS[name]} confidence must be between 0 and 1.")
        thresholds[name] = value
    return thresholds
