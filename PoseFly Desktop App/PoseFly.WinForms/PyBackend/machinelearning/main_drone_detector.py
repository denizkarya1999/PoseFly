from pathlib import Path
from ultralytics import YOLO

class DroneDetector:
    def __init__(self, model_path="models/Posefly.pt"):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        return results
