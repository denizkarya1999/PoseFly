from pathlib import Path
from ultralytics import YOLO

# Distance Labels
DISTANCE_LABELS = ["1m", "2m", "3m", "4m", "5m"]

class DistanceDetector:
    def __init__(self, model_path="models/Posefly_Distance.pt"):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        distance_data = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = DISTANCE_LABELS[cls]
            coords = box.xyxy[0].cpu().numpy()
            distance_data.append((coords, label, conf))

        return distance_data
