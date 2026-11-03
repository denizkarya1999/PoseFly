"""Regression coverage with real trained weights and labelled drone photographs."""
import itertools
import os
import time
import tempfile
import threading
from unittest.mock import Mock, patch
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'PoseFly Desktop App/PoseFly.WinForms/PyBackend'))
from drone_dataset import samples, iou


class DroneCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from machinelearning.main_drone_detector import DroneDetector
        cls.detector = DroneDetector()
        cls.detector.model.overrides['device'] = 'cpu'

    def test_real_drone_photographs_match_labels(self):
        for name, frame, truth in itertools.islice(samples(), 3):
            with self.subTest(image=name):
                result = self.detector.detect(frame)
                predictions = result.boxes.xyxy.cpu().numpy()
                for box in truth:
                    self.assertGreaterEqual(max((iou(box, pred) for pred in predictions), default=0), .5)

    def test_blank_frame_does_not_produce_a_drone(self):
        result = self.detector.detect(np.zeros((480, 640, 3), np.uint8))
        self.assertEqual(len(result.boxes), 0)


@unittest.skipUnless(os.environ.get('DISPLAY'), 'Run with xvfb-run for the actual Tk preview test')
class DronePreviewTests(unittest.TestCase):
    def test_real_model_boxes_reach_preview_after_slow_inference(self):
        import tkinter as tk
        from PIL import ImageTk
        import posefly_gui
        from backend import PipelineBackend
        from machinelearning.main_drone_detector import DroneDetector
        _, photograph, _ = next(samples())
        detector = DroneDetector()
        detector.model.overrides['device'] = 'cpu'
        backend = PipelineBackend()
        backend.cv.drone = detector
        backend.open_camera = Mock()
        backend.apply_camera_settings_led_id = Mock()
        backend.set_rollingshutter = Mock(return_value='Test camera')
        backend.read_frame = lambda: (True, photograph.copy())
        original_process = backend.process_frame
        def delayed_process(frame, toggles, on_detections=None):
            time.sleep(.7)  # Exceeds the old deadline before inference even begins.
            return original_process(frame, toggles, on_detections=on_detections)
        backend.process_frame = delayed_process
        root = tk.Tk()
        root.withdraw()
        with tempfile.TemporaryDirectory() as tmp, patch.object(posefly_gui, 'list_available_cameras', return_value=[]):
            app = posefly_gui.PoseFlyGUI(root)
            app.backend = backend
            app.save_var.set(False)
            app.out_path_var.set(str(Path(tmp) / 'preview.mp4'))
            for variable in (app.tog_angle, app.tog_distance, app.tog_led, app.tog_speed):
                variable.set(False)
            try:
                app.start()
                deadline = time.monotonic() + 20
                visible = False
                while time.monotonic() < deadline:
                    root.update()
                    if app._detections and app._tk_img is not None:
                        x1, y1, x2, y2, _ = app._detections[0]
                        pixels = np.asarray(ImageTk.getimage(app._tk_img))
                        visible = np.array_equal(pixels[y2, (x1 + x2) // 2, :3], [255, 180, 0])
                        if visible:
                            break
                    time.sleep(.01)
                self.assertTrue(visible, f'No model-generated box reached the preview: {app.status["text"]}')
            finally:
                app._stop_event.set()
                if app._thread:
                    app._thread.join(20)
                for callback in root.tk.call('after', 'info'):
                    root.after_cancel(callback)
                root.destroy()
