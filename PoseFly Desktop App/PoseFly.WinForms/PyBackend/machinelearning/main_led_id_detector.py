from pathlib import Path
import sys
import numpy as np
from ultralytics import YOLO
import math
import cv2

import pathlib, os
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

# Ensure models/ is on sys.path (YOLO YAML safe)
MODELS_DIR = Path(__file__).parent / "models/OutDoorModels/Modules"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

# Import + register LSA for Ultralytics YAML
from LSA import LSA
from SE import SE

import ultralytics.nn.tasks as tasks
tasks.LSA = LSA
tasks.SE = SE

# Labels for each LED
LED_LABELS = ["0001", "0101", "1110", "1001"]

# Rolling-shutter configuration
ISO = 800
SHUTTER_HZ = 6000

def apply_rolling_shutter(frame, iso=ISO, shutter_hz=SHUTTER_HZ):
    h, w, _ = frame.shape
    row_time = 1.0 / max(1, int(shutter_hz))
    out = frame.copy()

    # --- Camera-like mapping (matches camera.py) ---

    # ISO -> gain (log mapping)
    iso_f = float(max(1, iso))
    t_iso = (math.log(iso_f) - math.log(50.0)) / (math.log(6400.0) - math.log(50.0))
    t_iso = max(0.0, min(1.0, t_iso))
    gain = 2.0 + t_iso * 18.0  # ~2..20

    # shutter_hz -> exposure proxy (higher Hz => darker)
    sh_f = float(max(1, shutter_hz))
    t_sh = (math.log(sh_f) - math.log(5.0)) / (math.log(6000.0) - math.log(5.0))
    t_sh = max(0.0, min(1.0, t_sh))

    exposure = -10.0 + (1.0 - t_sh) * 6.0
    exposure_scale = 2.0 ** (exposure / 2.0)

    brightness = 95.0 + (1.0 - t_sh) * 15.0
    brightness_scale = brightness / 100.0

    # --- OOK stripe model (row-time based) ---
    led_freq_hz = 2000
    duty = 0.5
    contrast = 0.90

    y = np.arange(h, dtype=np.float32)
    t = y * row_time
    phase = (t * led_freq_hz) % 1.0
    on = (phase < duty).astype(np.float32)

    row_gain = (1.0 - contrast) + contrast * on
    row_gain = row_gain[:, None]

    # Apply everything in HSV value channel
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]

    v *= gain
    v *= exposure_scale
    v *= brightness_scale
    v *= row_gain

    hsv[:, :, 2] = np.clip(v, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out

class LEDDetector:
    def __init__(self, model_path="models/OutdoorModels/PoseFly_Outdoor_LED_ID_Detection_YOLO_v26m_with_Self_Attention_and_C3K2_SA.pt"):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.001, iou=0.7):
        # Apply rolling shutter before inference
        frame = apply_rolling_shutter(frame)

        r = self.model.predict(
            frame,
            verbose=False,
            conf=conf,
            iou=iou,
            max_det=1
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        b = r.boxes[0]
        cls = int(b.cls[0])
        c = float(b.conf[0])
        coords = b.xyxy[0].cpu().numpy()

        label = LED_LABELS[cls] if 0 <= cls < len(LED_LABELS) else str(cls)
        return [(coords, label, c)]