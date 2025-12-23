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

    def set_rollingshutter(self, iso: int, shutter_hz: float):
        self.camera.rollingshutter(iso, shutter_hz)

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
    def process_frame(self, frame, toggles):
        return self.cv.process(frame, toggles)