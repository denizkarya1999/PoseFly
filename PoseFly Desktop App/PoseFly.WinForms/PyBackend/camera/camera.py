# camera.py
import os
import cv2
import math


class Camera:
    def __init__(self):
        self.cap = None
        self.w = None
        self.h = None

        # ---- Rolling shutter / ISO state ----
        self.iso = 100
        self.shutter_hz = 1000.0

        self.out = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.last_writer_path = None
        self.last_writer_fps = None

    # -------- Camera --------
    def open(self, camera_index=0, use_dshow=True):
        if self.cap is not None:
            return

        self.cap = (
            cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if use_dshow else
            cv2.VideoCapture(camera_index)
        )

        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError("Could not open webcam.")

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Apply last-known settings on open
        self._apply_rollingshutter()

    def apply_led_settings(self):
        """
        Backwards-compatible function.
        Your backend/server calls this during START.
        We'll just apply the current rollingshutter state.
        """
        if self.cap is None:
            raise RuntimeError("Camera not opened.")
        self._apply_rollingshutter()

    def rollingshutter(self, iso: int, shutter_hz: float):
        """
        Set ISO-like + shutter rate (Hz) and apply.
        """
        self.iso = int(max(50, min(6400, iso)))
        self.shutter_hz = float(max(5.0, min(6000.0, shutter_hz)))
        self._apply_rollingshutter()

    def _apply_rollingshutter(self):
        """
        Apply stored ISO + shutter_hz to the camera.
        Note: webcams don't expose true rolling-shutter readout timing;
              this maps to exposure/gain/brightness as a practical control.
        """
        if self.cap is None:
            return

        # Prefer manual exposure (DirectShow convention: ~0.75 manual)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        # ISO -> gain (log mapping)
        t_iso = (math.log(self.iso) - math.log(50)) / (math.log(6400) - math.log(50))
        t_iso = max(0.0, min(1.0, t_iso))
        gain = 2.0 + t_iso * 18.0  # ~2..20
        self.cap.set(cv2.CAP_PROP_GAIN, float(gain))

        # shutter_hz -> exposure (higher Hz => shorter exposure => darker)
        t_sh = (math.log(self.shutter_hz) - math.log(5.0)) / (math.log(6000.0) - math.log(5.0))
        t_sh = max(0.0, min(1.0, t_sh))
        exposure = -10.0 + (1.0 - t_sh) * 6.0  # ~[-10..-4]
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

        # brightness compensation
        brightness = 95.0 + (1.0 - t_sh) * 15.0
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, float(brightness))

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # -------- Writer --------
    def _ensure_writer(self, path, fps):
        if not path:
            return

        if (
            self.out is not None and
            self.last_writer_path == path and
            self.last_writer_fps == fps
        ):
            return

        self.release_writer()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.out = cv2.VideoWriter(path, self.fourcc, fps, (self.w, self.h))
        if not self.out.isOpened():
            self.out = None
            raise RuntimeError("Failed to open VideoWriter.")

        self.last_writer_path = path
        self.last_writer_fps = fps

    def write_if_enabled(self, frame, enabled, path, fps):
        if not enabled:
            self.release_writer()
            return
        self._ensure_writer(path, fps)
        self.out.write(frame)

    def release_writer(self):
        if self.out is not None:
            self.out.release()
            self.out = None
        self.last_writer_path = None
        self.last_writer_fps = None