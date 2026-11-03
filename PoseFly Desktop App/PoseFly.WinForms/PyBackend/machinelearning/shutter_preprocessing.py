"""Artificial exposure/rolling-shutter effect for feature inference only.

This retains PoseFly's software approximation, using each model's training
ISO and shutter metadata. The fixed 2 kHz, 50% duty reference pattern is not
a measurement of the observed LED. Digital processing cannot recover sensor
timing, motion detail, or LED information absent from the captured image.
Nothing here changes camera controls or writes into the source frame.
"""

import math

import cv2
import numpy as np


def simulate_shutter(frame, *, iso, shutter_hz):
    """Return a separate uint8 BGR image with the model's artificial effect."""
    iso, shutter_hz = float(iso), float(shutter_hz)
    if not all(math.isfinite(value) and value > 0 for value in (iso, shutter_hz)):
        raise ValueError("Model ISO and shutter rate must be finite and positive.")
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
            or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0):
        raise ValueError("Feature preprocessing requires a nonempty uint8 BGR image.")

    # These are bounded software gain/exposure proxies, not camera control units.
    iso_fraction = np.clip(math.log(iso / 50.0) / math.log(6400.0 / 50.0), 0.0, 1.0)
    shutter_fraction = np.clip(math.log(shutter_hz / 5.0) / math.log(6000.0 / 5.0), 0.0, 1.0)
    gain = 2.0 + iso_fraction * 18.0
    exposure_scale = 2.0 ** ((-10.0 + (1.0 - shutter_fraction) * 6.0) / 2.0)
    brightness_scale = (95.0 + (1.0 - shutter_fraction) * 15.0) / 100.0

    # Artificial reference bands; shutter time is not the sensor's row-read time.
    rows = np.arange(frame.shape[0], dtype=np.float32)
    phase = (rows / shutter_hz * 2000.0) % 1.0
    row_gain = (0.1 + 0.9 * (phase < 0.5).astype(np.float32))[:, None]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(
        hsv[:, :, 2] * gain * exposure_scale * brightness_scale * row_gain,
        0, 255,
    )
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
