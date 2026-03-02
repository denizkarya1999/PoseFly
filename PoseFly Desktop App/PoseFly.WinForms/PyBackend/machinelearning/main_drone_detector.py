from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO
import sys

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

class DroneDetector:
    def __init__(self, model_path="models/OutdoorModels/PoseFly_Outdoor_Drone_Detection_YOLO_v26m_with_Self_Attention_and_C3K2_SA.pt"):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    # -------------------------------------------------
    # Resize: Fill (with center crop) → 512x512
    # -------------------------------------------------
    def resize_fill_512(self, image):
        target = 512
        h, w = image.shape[:2]

        # Scale so that the smaller side becomes 512
        scale = target / min(h, w)

        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Center crop to 512x512
        start_x = (new_w - target) // 2
        start_y = (new_h - target) // 2

        cropped = resized[start_y:start_y + target, start_x:start_x + target]

        return cropped

    def detect(self, frame, conf_threshold=0.1, iou=0.7, return_preprocessed=False):
        # Apply resize-fill preprocessing (optional)
        #frame = self.resize_fill_512(frame)

        # Run inference
        result = self.model.predict(
            frame,
            iou = iou,
            imgsz=512,
            conf=conf_threshold,
            verbose=False
        )[0]

        if return_preprocessed:
            return result, frame

        return result
