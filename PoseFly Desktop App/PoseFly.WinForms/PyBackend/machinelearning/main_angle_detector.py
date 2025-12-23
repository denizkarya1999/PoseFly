from pathlib import Path
from ultralytics import YOLO

ANGLE_LABELS = ["0_360°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"]

class AngleDetector:
    def __init__(self, model_path="models/Posefly_Angle.pt"):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame, conf=0.001, iou=0.7):
        """
        Detection-style model:
        - Use very low conf so we usually get at least 1 box
        - Return top-1 prediction as [(coords, label, conf)]
        """
        r = self.model.predict(frame, verbose=False, conf=conf, iou=iou, max_det=1)[0]

        if r.boxes is None or len(r.boxes) == 0:
            return []

        b = r.boxes[0]
        cls = int(b.cls[0])
        c = float(b.conf[0])
        coords = b.xyxy[0].cpu().numpy()

        label = ANGLE_LABELS[cls] if 0 <= cls < len(ANGLE_LABELS) else str(cls)
        return [(coords, label, c)]