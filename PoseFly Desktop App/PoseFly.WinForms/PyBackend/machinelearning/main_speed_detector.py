from ultralytics import YOLO
from pathlib import Path
import sys

import pathlib, os
if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

# Ensure models/ is on sys.path (YOLO YAML safe)
MODELS_DIR = Path(__file__).parent / "models/OutdoorModels/Modules"

if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

# Import + register LSA for Ultralytics YAML
from LSA import LSA
from SE import SE

import ultralytics.nn.tasks as tasks
tasks.LSA = LSA
tasks.SA = SE

SPEED_LABELS = ["Fast", "Slow", "Static"]

# Per-model settings apply only to an inference image, never to the camera.
from machinelearning.model_profiles import FEATURE_MODELS
from machinelearning.shutter_preprocessing import simulate_shutter
ISO = FEATURE_MODELS['speed'].iso
SHUTTER_HZ = FEATURE_MODELS['speed'].shutter_hz


class SpeedDetector:
    def __init__(self, model_path=FEATURE_MODELS['speed'].path):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.001, iou=0.7):
        model_frame = simulate_shutter(frame, iso=ISO, shutter_hz=SHUTTER_HZ)

        r = self.model.predict(
            model_frame,
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

        label = SPEED_LABELS[cls] if 0 <= cls < len(SPEED_LABELS) else str(cls)
        return [(coords, label, c)]
