from pathlib import Path
from ultralytics import YOLO

SPEED_LABELS = ["Static", "Fast", "Slow"]

class SpeedDetector:
    def __init__(self, model_path="models/PoseFly_Speed_Detection.pt"):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.001, iou=0.7):
        r = self.model.predict(frame, verbose=False, conf=conf, iou=iou, max_det=1)[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        b = r.boxes[0]
        cls = int(b.cls[0])
        c = float(b.conf[0])
        coords = b.xyxy[0].cpu().numpy()

        label = SPEED_LABELS[cls] if 0 <= cls < len(SPEED_LABELS) else str(cls)
        return [(coords, label, c)]