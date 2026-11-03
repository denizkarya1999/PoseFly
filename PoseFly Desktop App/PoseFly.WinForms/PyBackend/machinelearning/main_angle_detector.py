from pathlib import Path
import sys
from ultralytics import YOLO

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

# Labels for each angle.
ANGLE_LABELS = ["0_360", "135", "180", "225", "270", "315", "45", "90"]

# Per-model settings apply only to an inference image, never to the camera.
from machinelearning.model_profiles import FEATURE_MODELS
from machinelearning.confidence import DEFAULT_CONFIDENCE
from machinelearning.box_limits import DEFAULT_MAX_BOXES, parse_box_limit
from machinelearning.shutter_preprocessing import simulate_shutter
ISO = FEATURE_MODELS['angle'].iso
SHUTTER_HZ = FEATURE_MODELS['angle'].shutter_hz


class AngleDetector:
    def __init__(self, model_path=FEATURE_MODELS['angle'].path):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame, conf=DEFAULT_CONFIDENCE["angle"], iou=0.7,
               max_det=DEFAULT_MAX_BOXES["angle"]):
        """
        Detection-style model:
        - Respect the selected minimum confidence
        - Return up to the selected maximum number of predictions
        """
        max_det = parse_box_limit(max_det)
        model_frame = simulate_shutter(frame, iso=ISO, shutter_hz=SHUTTER_HZ)

        r = self.model.predict(
            model_frame,
            verbose=False,
            conf=conf,
            iou=iou,
            max_det=max_det
        )[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        predictions = []
        for b in r.boxes[:max_det]:
            cls = int(b.cls[0])
            c = float(b.conf[0])
            coords = b.xyxy[0].cpu().numpy()
            label = ANGLE_LABELS[cls] if 0 <= cls < len(ANGLE_LABELS) else str(cls)
            predictions.append((coords, label, c))
        return predictions
