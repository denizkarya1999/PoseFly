# camera.py
import os
import cv2


class Camera:
    def __init__(self):
        self.cap = None
        self.w = None
        self.h = None

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

    def apply_led_settings(self):
        if self.cap is None:
            raise RuntimeError("Camera not opened.")
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -5)
        self.cap.set(cv2.CAP_PROP_GAIN, 20)
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 150)

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