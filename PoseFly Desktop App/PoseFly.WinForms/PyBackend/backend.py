from camera.camera import Camera
from videoprocessing.computer_vision import ComputerVision


class PipelineBackend:
    def __init__(self):
        self.camera = Camera()
        self.cv = ComputerVision()

    # Camera
    def open_camera(self, *args, **kwargs):
        self.camera.open(*args, **kwargs)

    def apply_camera_settings_led_id(self):
        # server calls this; keep compatibility
        self.camera.apply_led_settings()

    def set_rollingshutter(self, iso: int, shutter_hz: float, enabled=True):
        return self.camera.rollingshutter(iso, shutter_hz, enabled)

    def read_frame(self):
        return self.camera.read()

    def release_camera(self):
        self.camera.release()

    # Writer
    def write_frame_if_enabled(self, frame, save, path, fps):
        self.camera.write_if_enabled(frame, save, path, fps)

    def release_writer(self):
        self.camera.release_writer()

    # Vision
    def process_frame(self, frame, toggles, on_detections=None):
        return self.cv.process(frame, toggles, on_detections=on_detections)

    def drone_detections(self):
        return self.cv.last_drone_detections

    def drone_features(self):
        return self.cv.last_drone_features
