# computer_vision.py
import time
import cv2
import os
import sys
import subprocess

from importlib import import_module

# Import and load only enabled models, on the pipeline worker rather than the UI.
DETECTORS = {
    "drone": ("main_drone_detector", "DroneDetector"),
    "angle": ("main_angle_detector", "AngleDetector"),
    "distance": ("main_distance_detector", "DistanceDetector"),
    "led": ("main_led_id_detector", "LEDDetector"),
    "speed": ("main_speed_detector", "SpeedDetector"),
}

DRONE_MODELS = {
    "Indoor": "models/IndoorModels/Posefly.pt",
    "Outdoor": "models/OutdoorModels/PoseFly_Outdoor_Drone_Detection_Original_YOLO_v26m.pt",
}


class ComputerVision:
    def __init__(self):
        for name in DETECTORS:
            setattr(self, name, None)

        self.drone_profile = "Indoor"
        self._drone_cache = {}

        # Logging control (rate-limited)
        self._last_log_t = 0.0
        self.log_every_sec = 0.5
        self.log_top_k = 8

        self._iter = 0
        self.last_drone_detections = []
        self.last_drone_features = []

        # Prevent accidental multiple launches
        self._trajectory_already_run = False

    def _detector(self, name):
        detector = getattr(self, name)
        if detector is None:
            module, class_name = DETECTORS[name]
            factory = getattr(import_module(f"machinelearning.{module}"), class_name)
            if name == "drone":
                detector = factory(model_path=DRONE_MODELS[self.drone_profile])
                self._drone_cache[self.drone_profile] = detector
            else:
                detector = factory()
            setattr(self, name, detector)
        return detector

    def _should_log_now(self) -> bool:
        """Return True if enough time has passed to print logs."""
        if self.log_every_sec <= 0:
            return True
        now = time.monotonic()
        if (now - self._last_log_t) >= self.log_every_sec:
            self._last_log_t = now
            return True
        return False

    # ---------- Unified extraction for iteration logging ----------

    def _items_from_list(self, out):
        # Output already in list format: [(coords, label, conf)]
        if not isinstance(out, list) or len(out) == 0:
            return None
        items = []
        for tup in out:
            if not isinstance(tup, (list, tuple)) or len(tup) < 3:
                continue
            lbl = str(tup[1])
            conf = float(tup[2])
            items.append((lbl, conf))
        return items if items else None

    def _items_from_detection(self, out):
        # Ultralytics Results format (YOLO-style)
        if out is None or not hasattr(out, "boxes") or out.boxes is None or len(out.boxes) == 0:
            return None
        names = getattr(out, "names", None)
        items = []
        for b in out.boxes:
            cls_id = int(b.cls[0]) if hasattr(b, "cls") else -1
            conf = float(b.conf[0])
            lbl = str(cls_id)
            if isinstance(names, dict) and cls_id in names:
                lbl = names[cls_id]
            items.append((lbl, conf))
        return items if items else None

    def _collect_items(self, out):
        # Try list-format first, fallback to YOLO detection format
        items = self._items_from_list(out)
        if items is None:
            items = self._items_from_detection(out)
        return items

    def _top_k_str(self, out):
        # Format top-K predictions for logging
        items = self._collect_items(out)
        if not items:
            return "(no)"
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        k = self.log_top_k
        shown = items_sorted if (k is None or k <= 0) else items_sorted[:k]
        parts = [f"{lbl} {p * 100.0:.1f}%" for (lbl, p) in shown]
        if len(shown) < len(items_sorted):
            parts.append(f"+{len(items_sorted) - len(shown)} more")
        return "(" + ", ".join(parts) + ")"

    def _best_item(self, out):
        """Return (label, confidence) of the highest-confidence prediction."""
        items = self._collect_items(out)
        if not items:
            return None, None
        lbl, conf = max(items, key=lambda x: x[1])
        return str(lbl), float(conf)

    def _print_iteration_line(self, drone_out=None, angle_out=None, dist_out=None, led_out=None, speed_out=None):
        # One-line summary per iteration
        self._iter += 1
        pieces = [f"Iteration-{self._iter}:"]

        if drone_out is not None:
            pieces.append(f"DRONE {self._top_k_str(drone_out)}")
        if angle_out is not None:
            pieces.append(f"ANGLE {self._top_k_str(angle_out)}")
        if dist_out is not None:
            pieces.append(f"DISTANCE {self._top_k_str(dist_out)}")
        if led_out is not None:
            pieces.append(f"LED {self._top_k_str(led_out)}")
        if speed_out is not None:
            pieces.append(f"SPEED {self._top_k_str(speed_out)}")

        print(", ".join(pieces))

    # ---------- Header label (top overlay) ----------

    @staticmethod
    def _wrap_by_pipe(text: str, max_chars: int):
        # Break long header text into at most two readable lines
        tokens = text.split(" | ")
        lines = []
        cur = ""
        for t in tokens:
            if not cur:
                cur = t
            elif len(cur) + 3 + len(t) <= max_chars:
                cur += " | " + t
            else:
                lines.append(cur)
                cur = t
        if cur:
            lines.append(cur)
        return lines

    @staticmethod
    def _draw_header_on_frame(frame_bgr, x1, y1, x2, header_text):
        # Draw black background + white text above the drone box
        H, W = frame_bgr.shape[:2]
        x1 = max(0, min(int(x1), W - 1))
        x2 = max(0, min(int(x2), W - 1))
        y1 = max(0, min(int(y1), H - 1))
        box_w = max(1, x2 - x1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        max_chars = max(20, int(box_w / 10))
        lines = ComputerVision._wrap_by_pipe(header_text, max_chars=max_chars)
        lines = lines[:2]  # limit to 2 lines

        line_h = 0
        for ln in lines:
            (_, th), _ = cv2.getTextSize(ln, font, font_scale, thickness)
            line_h = max(line_h, th)

        pad_y = 6
        strip_h = (line_h + pad_y) * len(lines) + pad_y

        y_top = y1 - strip_h - 2
        if y_top < 0:
            y_top = y1 + 2  # fallback: draw inside box
        y_bottom = min(H - 1, y_top + strip_h)

        cv2.rectangle(frame_bgr, (x1, y_top), (x2, y_bottom), (0, 0, 0), -1)

        y_text = y_top + pad_y + line_h
        for ln in lines:
            cv2.putText(frame_bgr, ln, (x1 + 6, y_text), font, font_scale, (255, 255, 255), thickness)
            y_text += (line_h + pad_y)

        return frame_bgr

    # ---------- Trajectory launcher ----------

    def run_trajectory_algorithm(self, wait=False, log_file_path=None):
        """
        Run trajectory_algorithm.py once after inference finishes.

        Args:
            wait: if True, block until script exits
            log_file_path: optional txt file path to pass to trajectory_algorithm.py
        """
        if self._trajectory_already_run:
            print("trajectory_algorithm.py was already launched once. Skipping.")
            return

        current_dir = os.path.dirname(os.path.abspath(__file__))
        tools_dir = os.path.join(current_dir, "tools")
        trajectory_script = os.path.join(tools_dir, "trajectory_algorithm.py")

        if not os.path.isfile(trajectory_script):
            print(f"Trajectory script not found: {trajectory_script}")
            return

        cmd = [sys.executable, trajectory_script]

        # Optional: pass log file path if your trajectory script supports argv
        if log_file_path:
            cmd.append(log_file_path)

        try:
            print(f"Launching trajectory algorithm: {' '.join(cmd)}")
            if wait:
                subprocess.run(cmd, check=True)
            else:
                subprocess.Popen(cmd)
            self._trajectory_already_run = True
        except Exception as e:
            print(f"Failed to launch trajectory_algorithm.py: {e}")

    # ---------- Main pipeline ----------

    def process(self, frame, toggles, on_detections=None):
        self.last_drone_detections = []
        self.last_drone_features = []
        # Feature estimates are meaningful only for detected drone crops.
        if not toggles.get("drone", True):
            if on_detections is not None:
                on_detections([])
            return frame
        profile = toggles.get("drone_model", "Indoor")
        if profile not in DRONE_MODELS:
            raise ValueError(f"Unknown drone model: {profile}")
        if profile != self.drone_profile:
            self.drone_profile = profile
            self.drone = self._drone_cache.get(profile)
        do_log = self._should_log_now()
        drone_out = self._detector("drone").detect(frame)
        boxes = getattr(drone_out, "boxes", None)
        if boxes is None or len(boxes) == 0:
            if on_detections is not None:
                on_detections([])
            if do_log:
                self._print_iteration_line(drone_out=drone_out)
            return frame

        h, w = frame.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            self.last_drone_detections.append((x1, y1, x2, y2, float(box.conf[0])))
        # Publish drone boxes before loading/running the slower feature models.
        if on_detections is not None:
            on_detections(list(self.last_drone_detections))
        for x1, y1, x2, y2, confidence in self.last_drone_detections:
            # Every submodel receives the same untouched camera crop.
            crop = frame[y1:y2, x1:x2].copy()
            outputs, features = {}, {}
            for name in ("angle", "distance", "led", "speed"):
                if toggles.get(name, True):
                    outputs[name] = self._detector(name).detect(crop)
                    features[name] = self._best_item(outputs[name])
            self.last_drone_features.append(features)
            if do_log:
                self._print_iteration_line(
                    drone_out=drone_out, angle_out=outputs.get("angle"),
                    dist_out=outputs.get("distance"), led_out=outputs.get("led"),
                    speed_out=outputs.get("speed"),
                )
        from videoprocessing.drone_overlay import draw_drones
        return draw_drones(frame, self.last_drone_detections, self.last_drone_features)
