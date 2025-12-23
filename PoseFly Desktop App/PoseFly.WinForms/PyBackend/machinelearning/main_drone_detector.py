from pathlib import Path
from ultralytics import YOLO

class DroneDetector:
    def __init__(self, model_path="models/Posefly.pt"):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame):
        # Detection model -> Results with .boxes
        results = self.model(frame, verbose=False)[0]
        return results