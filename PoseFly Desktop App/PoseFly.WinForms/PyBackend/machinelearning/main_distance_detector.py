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
tasks.SE = SE

# --------------------------------------------------
# YOLO import
# --------------------------------------------------
from ultralytics import YOLO

DISTANCE_LABELS = ["1m", "2m", "3m", "4m", "5m"]

# This checkpoint and its hardware capture preset share one definition.
from machinelearning.model_profiles import FEATURE_MODELS
ISO = FEATURE_MODELS['distance'].iso
SHUTTER_HZ = FEATURE_MODELS['distance'].shutter_hz


class DistanceDetector:
    def __init__(self, model_path=FEATURE_MODELS['distance'].path):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.001, iou=0.7):
        # Use the real camera exposure; do not synthesize stripes for inference.

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

        label = DISTANCE_LABELS[cls] if 0 <= cls < len(DISTANCE_LABELS) else str(cls)
        return [(coords, label, c)]