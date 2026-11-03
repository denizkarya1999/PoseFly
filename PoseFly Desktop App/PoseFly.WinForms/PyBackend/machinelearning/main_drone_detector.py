from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
from machinelearning.confidence import DEFAULT_CONFIDENCE
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

DEFAULT_MODEL = "models/IndoorModels/Posefly.pt"


class DroneDetector:
    def __init__(self, model_path=DEFAULT_MODEL):
        model_path = Path(__file__).parent / model_path
        if not model_path.is_file():
            raise FileNotFoundError(f"Drone model not found: {model_path}")
        self.model = YOLO(model_path)

    def detect(self, frame, conf_threshold=DEFAULT_CONFIDENCE["drone"], iou=0.7, return_preprocessed=False):
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                or frame.shape[2] != 3 or frame.size == 0 or frame.dtype != np.uint8):
            raise ValueError("Drone detection requires a nonempty uint8 BGR image.")
        # Let YOLO letterbox the entire image and return source-image coordinates.
        # A center crop excluded drones at the sides of widescreen camera frames.
        result = self.model.predict(
            frame, iou=iou, imgsz=640, conf=conf_threshold, verbose=False,
        )[0]
        if return_preprocessed:
            # Results and this image share the same original coordinate system.
            return result, frame
        return result
