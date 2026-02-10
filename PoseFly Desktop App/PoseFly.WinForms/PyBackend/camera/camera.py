# camera.py
import os
import cv2
import math
import numpy as np

# --- Brightness ---
BASE_BRIGHTNESS = 0.0


class Camera:
    def __init__(self):
        self.cap = None
        self.w = None
        self.h = None

        # ---- Rolling shutter / ISO state ----
        self.iso = 250
        self.shutter_hz = 1000.0

        # Precomputed row gain for OOK stripes (built in _apply_rollingshutter)
        self._ook_row_gain = None

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
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        # --- Camera-like mapping (matches apply_rolling_shutter) ---

        # ISO -> gain (log mapping)
        iso_f = float(max(1, self.iso))
        t_iso = (math.log(iso_f) - math.log(50.0)) / (math.log(6400.0) - math.log(50.0))
        t_iso = max(0.0, min(1.0, t_iso))
        gain = 2.0 + t_iso * 18.0  # ~2..20

        # shutter_hz -> exposure proxy (higher Hz => darker)
        sh_f = float(max(1, self.shutter_hz))
        t_sh = (math.log(sh_f) - math.log(5.0)) / (math.log(6000.0) - math.log(5.0))
        t_sh = max(0.0, min(1.0, t_sh))

        exposure = -10.0 + (1.0 - t_sh) * 6.0
        exposure_scale = 2.0 ** (exposure / 2.0)

        brightness = 95.0 + (1.0 - t_sh) * 15.0
        brightness_scale = brightness / 100.0

        # --- OOK stripe model (row-time based) ---
        led_freq_hz = 2000
        duty = 0.5
        contrast = 0.90

        row_time = 1.0 / max(1, int(self.shutter_hz))

        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if h > 0:
            y = np.arange(h, dtype=np.float32)
            t = y * row_time
            phase = (t * float(led_freq_hz)) % 1.0
            on = (phase < float(duty)).astype(np.float32)

            row_gain = (1.0 - float(contrast)) + float(contrast) * on
            self._ook_row_gain = row_gain[:, None]
        else:
            self._ook_row_gain = None

        # Apply what the camera can accept (global controls)
        self.cap.set(cv2.CAP_PROP_GAIN, float(gain))
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

        brightness_prop = BASE_BRIGHTNESS + 0.5 * (brightness_scale - 1.0)
        brightness_prop = max(0.0, min(1.0, brightness_prop))
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, float(brightness_prop))

        gamma = 1.0 / max(0.1, exposure_scale * brightness_scale)
        gamma = max(0.3, min(3.0, gamma))
        self.cap.set(cv2.CAP_PROP_GAMMA, float(gamma))

    def read(self):
        if self.cap is None:
            return False, None

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return ok, frame

        # Apply the same stripe effect used by apply_rolling_shutter()
        if self._ook_row_gain is not None:
            if self._ook_row_gain.shape[0] != frame.shape[0]:
                self._apply_rollingshutter()

            if self._ook_row_gain is not None and self._ook_row_gain.shape[0] == frame.shape[0]:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                v = hsv[:, :, 2]
                v *= self._ook_row_gain
                hsv[:, :, 2] = np.clip(v, 0, 255)
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return True, frame

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