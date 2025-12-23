# computer_vision.py
import time
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
        return items

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

    def _print_iteration_line(self, drone_out=None, angle_out=None, dist_out=None, led_out=None):
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

        print(", ".join(pieces))

    # ---------- Drawing helper for sub-model boxes on the crop ----------

    @staticmethod
    def _draw_list_boxes_on_crop(crop_bgr, out_list, title, color_bgr):
        """
        out_list: [(coords_xyxy, label, conf), ...] where coords are relative to this crop.
        Draws boxes + label + conf on crop and returns it.
        """
        if not isinstance(out_list, list) or len(out_list) == 0:
            return crop_bgr

        h, w = crop_bgr.shape[:2]
        for coords, lbl, conf in out_list:
            try:
                x1, y1, x2, y2 = coords
            except Exception:
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # clamp to crop bounds
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            cv2.rectangle(crop_bgr, (x1, y1), (x2, y2), color_bgr, 2)
            text = f"{title}: {lbl} ({float(conf):.2f})"
            cv2.putText(
                crop_bgr, text, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2
            )

        return crop_bgr

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

                if do_log:
                    self._print_iteration_line(
                        drone_out=drone_out if toggles.get("drone", True) else None,
                        angle_out=angle_out if toggles.get("angle", True) else None,
                        dist_out=dist_out if toggles.get("distance", True) else None,
                        led_out=led_out if toggles.get("led", True) else None,
                    )
            return frame

        # 3) With drones: per-drone crop processing + drawing
        for box in drone_out.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Drone box on full frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Drone", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                continue

            # Run sub-models on crop
            angle_out = self.angle.detect(crop) if toggles.get("angle", True) else None
            dist_out  = self.distance.detect(crop) if toggles.get("distance", True) else None
            led_out   = self.led.detect(crop) if toggles.get("led", True) else None

            # Draw their boxes ON THE CROP (relative coords)
            if toggles.get("angle", True):
                crop = self._draw_list_boxes_on_crop(crop, angle_out, "Angle", (0, 255, 0))
            if toggles.get("distance", True):
                crop = self._draw_list_boxes_on_crop(crop, dist_out, "Distance", (255, 0, 0))
            if toggles.get("led", True):
                crop = self._draw_list_boxes_on_crop(crop, led_out, "LED", (0, 0, 255))

            # Paste crop back into full frame
            frame[y1:y2, x1:x2] = crop

            # One-line iteration log (per drone box processed)
            if do_log:
                self._print_iteration_line(
                    drone_out=drone_out if toggles.get("drone", True) else None,
                    angle_out=angle_out if toggles.get("angle", True) else None,
                    dist_out=dist_out if toggles.get("distance", True) else None,
                    led_out=led_out if toggles.get("led", True) else None,
                )

        return frame