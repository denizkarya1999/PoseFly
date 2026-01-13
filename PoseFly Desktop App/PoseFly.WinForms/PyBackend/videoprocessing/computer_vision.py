# computer_vision.py
import time
import cv2

from machinelearning.main_drone_detector import DroneDetector
from machinelearning.main_angle_detector import AngleDetector
from machinelearning.main_distance_detector import DistanceDetector
from machinelearning.main_led_id_detector import LEDDetector
from machinelearning.main_speed_detector import SpeedDetector


class ComputerVision:
    def __init__(self):
        self.drone = DroneDetector()
        self.angle = AngleDetector()
        self.distance = DistanceDetector()
        self.led = LEDDetector()
        self.speed = SpeedDetector()  # expects 640x640 input

        # Logging control
        self._last_log_t = 0.0
        self.log_every_sec = 0.5
        self.log_top_k = 8
        self.run_submodels_on_full_frame_when_no_drone = True

        self._iter = 0

    def _should_log_now(self) -> bool:
        if self.log_every_sec <= 0:
            return True
        now = time.time()
        if (now - self._last_log_t) >= self.log_every_sec:
            self._last_log_t = now
            return True
        return False

    # ---------- Preprocess helpers ----------

    @staticmethod
    def _resize_640(bgr):
        return cv2.resize(bgr, (640, 640), interpolation=cv2.INTER_LINEAR)

    # ---------- Unified extraction for iteration logging ----------

    def _items_from_list(self, out):
        # list format: [(coords, label, conf)] OR [(id, label, conf)]
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
        # Ultralytics detection Results -> list[(class_label, conf)]
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
        items = self._items_from_list(out)
        if items is None:
            items = self._items_from_detection(out)
        return items

    def _top_k_str(self, out):
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
        """Return (label, conf) for the single best prediction, or (None, None)."""
        items = self._collect_items(out)
        if not items:
            return None, None
        lbl, conf = max(items, key=lambda x: x[1])
        return str(lbl), float(conf)

    def _print_iteration_line(self, drone_out=None, angle_out=None, dist_out=None, led_out=None, speed_out=None):
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
        lines = lines[:2]  # max 2 lines

        line_h = 0
        for ln in lines:
            (_, th), _ = cv2.getTextSize(ln, font, font_scale, thickness)
            line_h = max(line_h, th)
        pad_y = 6
        strip_h = (line_h + pad_y) * len(lines) + pad_y

        y_top = y1 - strip_h - 2
        if y_top < 0:
            y_top = y1 + 2  # inside box
        y_bottom = min(H - 1, y_top + strip_h)

        cv2.rectangle(frame_bgr, (x1, y_top), (x2, y_bottom), (0, 0, 0), -1)

        y_text = y_top + pad_y + line_h
        for ln in lines:
            cv2.putText(frame_bgr, ln, (x1 + 6, y_text), font, font_scale, (255, 255, 255), thickness)
            y_text += (line_h + pad_y)

        return frame_bgr

    # ---------- Main pipeline ----------

    def process(self, frame, toggles):
        do_log = self._should_log_now()

        # 1) Drone detection
        drone_out = self.drone.detect(frame) if toggles.get("drone", True) else None

        has_boxes = (
            drone_out is not None and
            hasattr(drone_out, "boxes") and
            drone_out.boxes is not None and
            len(drone_out.boxes) > 0
        )

        # 2) If no drones, optionally run sub-models on full frame for logging only
        if not has_boxes:
            if self.run_submodels_on_full_frame_when_no_drone:
                angle_out = self.angle.detect(frame) if toggles.get("angle", True) else None
                dist_out  = self.distance.detect(frame) if toggles.get("distance", True) else None
                led_out   = self.led.detect(frame) if toggles.get("led", True) else None
                speed_out = self.speed.detect(self._resize_640(frame)) if toggles.get("speed", True) else None

                if do_log:
                    self._print_iteration_line(
                        drone_out=drone_out if toggles.get("drone", True) else None,
                        angle_out=angle_out if toggles.get("angle", True) else None,
                        dist_out=dist_out if toggles.get("distance", True) else None,
                        led_out=led_out if toggles.get("led", True) else None,
                        speed_out=speed_out if toggles.get("speed", True) else None,
                    )
            return frame

        # 3) With drones: per-drone crop processing
        for b in drone_out.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            drone_conf = float(b.conf[0]) if hasattr(b, "conf") else 0.0

            # ONLY drone bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            # Run sub-models on crop (NO sub-box drawing)
            angle_out = self.angle.detect(crop) if toggles.get("angle", True) else None
            dist_out  = self.distance.detect(crop) if toggles.get("distance", True) else None
            led_out   = self.led.detect(crop) if toggles.get("led", True) else None
            speed_out = self.speed.detect(self._resize_640(crop)) if toggles.get("speed", True) else None

            # Build ONE header label at the top of the drone box
            led_lbl, led_conf = self._best_item(led_out)
            ang_lbl, ang_conf = self._best_item(angle_out)
            dst_lbl, dst_conf = self._best_item(dist_out)
            spd_lbl, spd_conf = self._best_item(speed_out)

            led_part = "Drone ID: N/A"
            ang_part = "Angle: N/A"
            dst_part = "Distance: N/A"
            spd_part = "Speed: N/A"

            if led_lbl is not None:
                led_part = f"Drone ID: {led_lbl}, {led_conf * 100.0:.1f}%"
            if ang_lbl is not None:
                ang_part = f"Angle: {ang_lbl}, {ang_conf * 100.0:.1f}%"
            if dst_lbl is not None:
                dst_part = f"Distance: {dst_lbl}, {dst_conf * 100.0:.1f}%"
            if spd_lbl is not None:
                spd_part = f"Speed: {spd_lbl}, {spd_conf * 100.0:.1f}%"

            header = (
                f"Drone: {drone_conf * 100.0:.1f}% | "
                f"{led_part} | {ang_part} | {dst_part} | {spd_part}"
            )
            self._draw_header_on_frame(frame, x1, y1, x2, header)

            if do_log:
                self._print_iteration_line(
                    drone_out=drone_out if toggles.get("drone", True) else None,
                    angle_out=angle_out if toggles.get("angle", True) else None,
                    dist_out=dist_out if toggles.get("distance", True) else None,
                    led_out=led_out if toggles.get("led", True) else None,
                    speed_out=speed_out if toggles.get("speed", True) else None,
                )

        return frame