from pathlib import Path
from ultralytics import YOLO

ANGLE_LABELS = ["0_360°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"]

class AngleDetector:
    def __init__(self, model_path="models/Posefly_Angle.pt"):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        angle_data = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            angle_label = ANGLE_LABELS[cls]
            coords = box.xyxy[0].cpu().numpy()
            angle_data.append((coords, angle_label, conf))

        return angle_data
