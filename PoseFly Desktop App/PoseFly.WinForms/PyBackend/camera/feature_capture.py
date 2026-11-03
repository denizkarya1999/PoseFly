"""Ask the capture worker for a real frame at a model's training exposure."""
from dataclasses import dataclass, field
import queue
import threading

from machinelearning.model_profiles import FEATURE_MODELS

# Discard buffered/transition frames after changing hardware exposure.
SETTLE_FRAMES = 3


@dataclass
class CaptureRequest:
    name: str
    ready: threading.Event = field(default_factory=threading.Event)
    frame: object = None


class FeatureCapture:
    def __init__(self, stop, wake):
        self.stop = stop
        self.wake = wake
        self.last_report = None
        self.requests = queue.SimpleQueue()
        self.exposure_changed = False

    def pending(self):
        return not self.requests.empty()

    def request(self, name):
        """Called by inference. Camera operations remain on the capture worker."""
        if self.stop.is_set():
            return None
        request = CaptureRequest(name)
        self.requests.put(request)
        self.wake.set()
        while not request.ready.wait(.05):
            if self.stop.is_set():
                return None
        return request.frame

    def serve(self, backend):
        """Called only by the capture worker; returns whether exposure changed."""
        try:
            request = self.requests.get_nowait()
        except queue.Empty:
            return False
        try:
            if self.stop.is_set():
                return False
            preset = FEATURE_MODELS[request.name]
            self.exposure_changed = True
            # Hardware readback stays internal; the UI only reports drone exposure.
            self.last_report = str(backend.set_rollingshutter(preset.iso, preset.shutter_hz, enabled=True))
            for index in range(SETTLE_FRAMES + 1):
                if self.stop.is_set():
                    return True
                ok, frame = backend.read_frame()
                if not ok or frame is None:
                    raise RuntimeError("Camera stopped while capturing a feature frame.")
                if index == SETTLE_FRAMES:
                    request.frame = frame.copy()
            return True
        finally:
            request.ready.set()
