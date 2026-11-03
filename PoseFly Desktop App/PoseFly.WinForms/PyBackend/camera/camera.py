import os
import cv2
import math
import numpy as np
import platform


class Camera:
    def __init__(self):
        self.cap = None
        self.w = self.h = None
        self.camera_index = 0
        self.iso = 250
        self.shutter_hz = 1000.0
        self.out = None
        self.last_writer_path = self.last_writer_fps = self._writer_size = None

    @staticmethod
    def _safe_set(cap, prop, value):
        try:
            return bool(cap.set(prop, value))
        except Exception:
            return False

    def _set_auto_exposure(self, enabled):
        try:
            is_v4l2 = self.cap.getBackendName() == "V4L2"
        except (AttributeError, cv2.error):
            is_v4l2 = platform.system() == "Linux"
        value = (3 if enabled else 1) if is_v4l2 else (0.75 if enabled else 0.25)
        return self._safe_set(self.cap, cv2.CAP_PROP_AUTO_EXPOSURE, value)

    # ---------------- Camera lifecycle ----------------
    def open(self, camera_index=0, use_dshow=True):
        if self.cap is not None:
            return

        self.camera_index = int(camera_index)
        sysname = platform.system()

        # ---- COMPATIBILITY: choose a backend sequence per OS ----
        if sysname == "Windows":
            backends = ([cv2.CAP_DSHOW] if use_dshow else []) + [cv2.CAP_MSMF, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_V4L2, cv2.CAP_ANY] if sysname == "Linux" else [cv2.CAP_ANY]

        last_backend = None
        cap = None
        for b in backends:
            last_backend = b
            try:
                cap = cv2.VideoCapture(int(camera_index), b)
            except Exception:
                cap = None

            if cap is not None and cap.isOpened():
                self.cap = cap
                break

            try:
                if cap is not None:
                    cap.release()
            except Exception:
                pass

        if self.cap is None:
            raise RuntimeError(
                f"Could not open webcam (index={camera_index}) on {sysname}. Tried backends={backends}."
            )

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # V4L2 uses exposure enum values; DirectShow uses normalized values.
        self._set_auto_exposure(True)
        if sysname == "Linux":
            self._safe_set(self.cap, cv2.CAP_PROP_BUFFERSIZE, 1)

    def apply_led_settings(self):
        # Compatibility hook: settings are applied explicitly on the capture worker.
        if self.cap is None:
            raise RuntimeError("Camera not opened.")

    def rollingshutter(self, iso: int, shutter_hz: float, enabled=True):
        if not math.isfinite(float(shutter_hz)) or float(shutter_hz) <= 0:
            raise ValueError("Shutter reciprocal must be finite and positive.")
        if not math.isfinite(float(iso)) or float(iso) < 1:
            raise ValueError("ISO must be finite and positive.")
        # Preserve the user's request; the driver applies its actual limits.
        self.iso = int(iso)
        self.shutter_hz = float(shutter_hz)
        if self.cap is None:
            return "Camera not opened"
        if platform.system() == "Linux":
            # Import here so Windows can still import and use Camera.
            try:
                from .linux_controls import apply_controls
            except ImportError:  # Standalone camera/Tools/recorder.py
                from linux_controls import apply_controls
            return apply_controls(self.camera_index, self.iso, self.shutter_hz, enabled)
        if not enabled:
            return "Auto exposure" if self._set_auto_exposure(True) else "Auto exposure unsupported"
        # Only DirectShow's log2(seconds) mapping is known here.
        if self.cap.getBackendName() != "DSHOW":
            return "Manual exposure/ISO controls are unsupported by this backend"
        if not self._set_auto_exposure(False):
            return "Manual exposure unsupported"
        exposure = round(math.log2(1 / self.shutter_hz))
        ok = self._safe_set(self.cap, cv2.CAP_PROP_EXPOSURE, exposure)
        return ("Hardware exposure applied" if ok else "Exposure rejected by camera") + " | ISO unsupported"

    def read(self):
        # Preserve the real sensor output: no simulated stripes or brightness correction.
        return self.cap.read() if self.cap is not None else (False, None)

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.w = self.h = None

    # ---------------- Writer ----------------
    def _ensure_writer(self, path, fps, size):
        if not path or not str(path).strip():
            raise ValueError("A video output path is required.")
        path = os.fspath(path)
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Recording FPS must be a finite positive number.")
        if self.out is not None and self.last_writer_path == path:
            if self._writer_size != size or self.last_writer_fps != fps:
                raise ValueError("Frame size or FPS changed during recording; choose a new output file.")
            return
        self.release_writer()
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        codec = "MJPG" if path.lower().endswith(".avi") else "mp4v"
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"Failed to open video output: {path}")
        self.out = writer
        self.last_writer_path = path
        self.last_writer_fps = fps
        self._writer_size = size

    def write_if_enabled(self, frame, enabled, path, fps):
        if not enabled:
            self.release_writer()
            return
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Recording requires a uint8 BGR frame.")
        h, w = frame.shape[:2]
        if not h or not w:
            raise ValueError("Cannot record an empty frame.")
        self._ensure_writer(path, fps, (w, h))
        self.out.write(frame)

    def release_writer(self):
        if self.out is not None:
            self.out.release()
            self.out = None
        self.last_writer_path = None
        self.last_writer_fps = None
        self._writer_size = None
