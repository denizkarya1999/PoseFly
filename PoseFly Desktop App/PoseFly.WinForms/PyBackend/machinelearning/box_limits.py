"""Per-model maximum detection counts, independent of the ML runtime."""

import math

from machinelearning.confidence import MODEL_LABELS

MAX_BOX_LIMIT = 300
# Preserve Ultralytics' previous drone default and the feature top-1 defaults.
DEFAULT_MAX_BOXES = {"drone": 300, "angle": 1, "distance": 1, "led": 1, "speed": 1}


def parse_box_limit(value):
    """Accept only whole detection counts in the supported range."""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"Maximum boxes must be a whole number from 1 to {MAX_BOX_LIMIT}.") from None
    if (isinstance(value, bool) or not math.isfinite(number)
            or not number.is_integer() or not 1 <= number <= MAX_BOX_LIMIT):
        raise ValueError(f"Maximum boxes must be a whole number from 1 to {MAX_BOX_LIMIT}.")
    return int(number)


def box_limits(values=None):
    """Resolve partial settings without modifying defaults or caller data."""
    limits = DEFAULT_MAX_BOXES.copy()
    if values is not None:
        for name in MODEL_LABELS:
            limits[name] = parse_box_limit(values.get(name, limits[name]))
    return limits
