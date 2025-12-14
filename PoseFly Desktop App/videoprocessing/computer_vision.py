# computer_vision.py
import cv2

from machinelearning.main_drone_detector import DroneDetector
from machinelearning.main_angle_detector import AngleDetector
from machinelearning.main_distance_detector import DistanceDetector
from machinelearning.main_led_id_detector import LEDDetector


class ComputerVision:
    def __init__(self):
        self.drone = DroneDetector()
        self.angle = AngleDetector()
        self.distance = DistanceDetector()
        self.led = LEDDetector()

    def process(self, frame, toggles):
        if not toggles.get("drone", True):
            return frame

        results = self.drone.detect(frame)

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Drone", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            angle = self.angle.detect(crop) if toggles.get("angle", True) else []
            dist  = self.distance.detect(crop) if toggles.get("distance", True) else []
            led   = self.led.detect(crop) if toggles.get("led", True) else []

            tx, ty = x2 + 10, y1
            lh = 25

            if angle:
                _, lbl, c = angle[0]
                cv2.putText(frame, f"Angle: {lbl} ({c:.2f})",
                            (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                ty += lh

            if dist:
                _, lbl, c = dist[0]
                cv2.putText(frame, f"Distance: {lbl} ({c:.2f})",
                            (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                ty += lh

            if led:
                _, lbl, c = led[0]
                cv2.putText(frame, f"LED ID: {lbl} ({c:.2f})",
                            (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame