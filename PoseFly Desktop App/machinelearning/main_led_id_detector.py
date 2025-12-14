from pathlib import Path
from ultralytics import YOLO

# LED ID classes
LED_LABELS = ["0001", "0101", "1110", "1001"]

class LEDDetector:
    def __init__(self, model_path="models/Posefly_Led_id.pt"):
        model_path = Path(__file__).parent / model_path
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        led_data = []

        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = LED_LABELS[cls]
            coords = box.xyxy[0].cpu().numpy()
            led_data.append((coords, label, conf))

        return led_data