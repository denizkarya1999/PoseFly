from pathlib import Path
import numpy as np
from ultralytics import YOLO

class DroneDetector:
    def __init__(self, model_path="models/Posefly.pt"):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame):
        # Apply rolling shutter before inference
        return self.model(frame, verbose=False)[0]