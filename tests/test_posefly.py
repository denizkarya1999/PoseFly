"""Hardware-free regression tests; run from the repository root with unittest."""
import os
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

BACKEND = Path(__file__).resolve().parents[1] / 'PoseFly Desktop App/PoseFly.WinForms/PyBackend'
sys.path.insert(0, str(BACKEND))
from camera.camera import Camera
from camera import discovery
from videoprocessing.computer_vision import ComputerVision


class LatestFrameTests(unittest.TestCase):
    def test_slow_consumer_receives_newest_only(self):
        from latest_frame import LatestFrame
        mailbox = LatestFrame()
        for index in range(100):
            mailbox.put(index)
        self.assertEqual(mailbox.get(), 99)
        mailbox.close()
        self.assertIsNone(mailbox.get())

    def test_close_wakes_waiting_consumer_and_discards_pending_frames(self):
        from latest_frame import LatestFrame
        mailbox = LatestFrame()
        received = []
        worker = threading.Thread(target=lambda: received.append(mailbox.get()))
        worker.start()
        mailbox.close()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(received, [None])
        mailbox.put('ignored')
        self.assertIsNone(mailbox.get())


class DiscoveryTests(unittest.TestCase):
    def test_linux_sparse_indices_and_no_arbitrary_limit(self):
        with patch.object(discovery.os, 'listdir', return_value=['video42', 'video2', 'videoX', 'audio0']):
            self.assertEqual(discovery._linux_video_indices(), [2, 42])
            self.assertEqual(discovery._linux_video_indices(20), [2])

    def test_only_readable_devices_are_returned_and_released(self):
        good, metadata = Mock(), Mock()
        good.read.return_value = (True, np.zeros((48, 64, 3), np.uint8))
        metadata.read.return_value = (False, None)
        with patch.object(discovery.platform, 'system', return_value='Linux'), \
             patch.object(discovery, '_linux_video_indices', return_value=[2, 42]), \
             patch.object(discovery.cv2, 'VideoCapture', side_effect=[good, metadata]):
            self.assertEqual([c['index'] for c in discovery.list_available_cameras()], [2])
        good.release.assert_called_once()
        metadata.release.assert_called_once()


class CameraTests(unittest.TestCase):
    def test_v4l2_exposure_and_buffer_and_release(self):
        cap = Mock()
        cap.get.return_value = 640
        cap.getBackendName.return_value = 'V4L2'
        with patch('camera.camera.platform.system', return_value='Linux'), \
             patch('camera.camera.cv2.VideoCapture', return_value=cap):
            camera = Camera()
            camera.open(42)
            cap.set.assert_any_call(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
            cap.set.assert_any_call(cv2.CAP_PROP_BUFFERSIZE, 1)
            camera._set_auto_exposure(False)
            cap.set.assert_any_call(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            camera.release()
        cap.release.assert_called_once()

    def test_capture_pixels_are_not_modified(self):
        camera = Camera()
        camera.cap = Mock()
        frame = np.full((48, 64, 3), 125, np.uint8)
        camera.cap.read.return_value = (True, frame)
        ok, output = camera.read()
        self.assertTrue(ok)
        self.assertIs(output, frame)
        self.assertTrue((output == 125).all())

    def test_linux_controls_are_applied_to_selected_device(self):
        camera = Camera()
        camera.cap = Mock()
        camera.camera_index = 42
        with patch('camera.camera.platform.system', return_value='Linux'), \
             patch('camera.linux_controls.apply_controls', return_value='ISO unsupported') as apply:
            self.assertEqual(camera.rollingshutter(250, 1000), 'ISO unsupported')
            apply.assert_called_once_with(42, 250, 1000., True)
            camera.rollingshutter(250, 1000, enabled=False)
            self.assertEqual(apply.call_args.args, (42, 250, 1000., False))

    def test_record_actual_frame_size_and_decode_mp4_and_avi(self):
        # Capture metadata can be stale; the actual frame must set writer size.
        for extension in ('mp4', 'avi'):
            with self.subTest(extension=extension), tempfile.TemporaryDirectory() as tmp:
                camera = Camera()
                camera.w, camera.h = 1920, 1080
                path = str(Path(tmp) / f'recording.{extension}')
                for i in range(4):
                    camera.write_if_enabled(np.full((48, 64, 3), i * 30, np.uint8), True, path, 15)
                camera.release_writer()
                cap = cv2.VideoCapture(path)
                try:
                    count = 0
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        self.assertEqual(frame.shape, (48, 64, 3))
                        count += 1
                    self.assertEqual(count, 4)
                finally:
                    cap.release()

    def test_invalid_output_does_not_open_writer(self):
        camera = Camera()
        frame = np.zeros((48, 64, 3), np.uint8)
        with patch('camera.camera.cv2.VideoWriter') as writer:
            for path, fps in [('', 30), ('out.mp4', float('nan')), ('out.mp4', 0)]:
                with self.assertRaises(ValueError):
                    camera.write_if_enabled(frame, True, path, fps)
            writer.assert_not_called()

    def test_size_change_does_not_truncate_existing_recording(self):
        camera = Camera()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'out.mp4')
            camera.write_if_enabled(np.zeros((48, 64, 3), np.uint8), True, path, 15)
            with self.assertRaises(ValueError):
                camera.write_if_enabled(np.zeros((64, 64, 3), np.uint8), True, path, 15)
            camera.release_writer()

    def test_failed_writer_is_released(self):
        writer = Mock()
        writer.isOpened.return_value = False
        with patch('camera.camera.cv2.VideoWriter', return_value=writer):
            with self.assertRaises(RuntimeError):
                Camera().write_if_enabled(np.zeros((48, 64, 3), np.uint8), True, 'out.mp4', 15)
        writer.release.assert_called_once()


class VisionTests(unittest.TestCase):
    def test_startup_and_disabled_pipeline_do_not_import_ml(self):
        result = subprocess.run([sys.executable, '-c',
            'import sys; from backend import PipelineBackend; b=PipelineBackend(); '
            'assert "torch" not in sys.modules; assert "ultralytics" not in sys.modules'],
            cwd=BACKEND, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        vision = ComputerVision()
        frame = np.zeros((48, 64, 3), np.uint8)
        with patch('videoprocessing.computer_vision.import_module') as importer:
            vision.process(frame, dict.fromkeys(('drone', 'angle', 'distance', 'led', 'speed'), False))
            importer.assert_not_called()

    def test_enabled_model_is_loaded_once(self):
        vision = ComputerVision()
        module = Mock()
        module.AngleDetector.return_value.detect.return_value = []
        with patch('videoprocessing.computer_vision.import_module', return_value=module):
            for _ in range(2):
                vision._detector('angle').detect(None)
        module.AngleDetector.assert_called_once()

    def test_drone_only_does_not_load_secondary_models_and_shows_na(self):
        vision = ComputerVision()
        vision.drone = Mock()
        vision.drone.detect.return_value = Mock(boxes=[Mock(xyxy=[[10, 20, 60, 70]], conf=[.9])])
        vision._last_log_t = time.monotonic()
        frame = np.zeros((80, 100, 3), np.uint8)
        with patch('videoprocessing.computer_vision.import_module') as importer, \
             patch('videoprocessing.drone_overlay.cv2.putText', wraps=cv2.putText) as text:
            output = vision.process(frame, dict(drone=True, angle=False, distance=False, led=False, speed=False))
            importer.assert_not_called()
            self.assertEqual([call.args[1] for call in text.call_args_list],
                             ['Drone 90%', 'Angle: N/A', 'Distance: N/A', 'LED ID: N/A', 'Speed: N/A'])
            self.assertEqual(vision.last_drone_detections, [(10, 20, 60, 70, .9)])
            self.assertEqual(output.shape, frame.shape)
            self.assertFalse(frame.any())
        vision.drone.detect.return_value = Mock(boxes=[])
        vision.process(frame, dict(drone=True, angle=False, distance=False, led=False, speed=False))
        self.assertEqual(vision.last_drone_detections, [])

    def test_no_drone_does_not_load_or_run_feature_models(self):
        vision = ComputerVision()
        vision.drone = Mock()
        vision.drone.detect.return_value = Mock(boxes=[])
        frame = np.zeros((80, 100, 3), np.uint8)
        vision._last_log_t = time.monotonic()
        with patch('videoprocessing.computer_vision.import_module') as importer:
            self.assertIs(vision.process(frame, {}), frame)
            importer.assert_not_called()
        self.assertEqual(vision.last_drone_features, [])
        vision.drone.detect.reset_mock()
        vision.process(frame, dict(drone=False, angle=True, distance=True, led=True, speed=True))
        vision.drone.detect.assert_not_called()

    def test_all_features_stay_associated_with_their_drone(self):
        vision = ComputerVision()
        frame = np.zeros((480, 640, 3), np.uint8)
        vision.drone = Mock()
        vision.drone.detect.return_value = Mock(boxes=[
            Mock(xyxy=[[10, 20, 60, 70]], conf=[.9]),
            Mock(xyxy=[[100, 120, 160, 180]], conf=[.8])])
        for name, values in {'angle': ['45', '90'], 'distance': ['2m', '4m'],
                             'led': ['A', 'B'], 'speed': ['1m/s', '3m/s']}.items():
            detector = Mock()
            detector.detect.side_effect = [[(None, value, .85)] for value in values]
            setattr(vision, name, detector)
        vision._last_log_t = time.monotonic()
        with patch('videoprocessing.drone_overlay.cv2.putText', wraps=cv2.putText) as put_text:
            output = vision.process(frame, {})
        self.assertEqual(vision.last_drone_features[0]['led'], ('A', .85))
        self.assertEqual(vision.last_drone_features[1]['distance'], ('4m', .85))
        labels = [call.args[1] for call in put_text.call_args_list]
        for title in ('Angle:', 'Distance:', 'LED ID:', 'Speed:'):
            self.assertEqual(sum(text.startswith(title) for text in labels), 2)
        self.assertTrue(output.any())
        self.assertFalse(frame.any())
        vision.drone.detect.return_value = Mock(boxes=[])
        vision.process(frame, {})
        self.assertEqual(vision.last_drone_detections, [])
        self.assertEqual(vision.last_drone_features, [])

    def test_disabled_feature_does_not_load_and_missing_prediction_is_explicit(self):
        from videoprocessing.drone_overlay import feature_lines
        vision = ComputerVision()
        vision.drone = Mock()
        vision.drone.detect.return_value = Mock(boxes=[Mock(xyxy=[[1, 1, 50, 50]], conf=[.9])])
        vision.angle = Mock()
        vision.angle.detect.return_value = []
        vision._last_log_t = time.monotonic()
        with patch('videoprocessing.computer_vision.import_module') as importer:
            vision.process(np.zeros((80, 100, 3), np.uint8),
                           dict(drone=True, angle=True, distance=False, led=False, speed=False))
            importer.assert_not_called()
        self.assertEqual(feature_lines(vision.last_drone_features[0]),
                         ['Angle: N/A', 'Distance: N/A', 'LED ID: N/A', 'Speed: N/A'])

    def test_feature_models_use_natural_camera_pixels(self):
        from machinelearning.main_angle_detector import AngleDetector
        from machinelearning.main_distance_detector import DistanceDetector
        from machinelearning.main_led_id_detector import LEDDetector
        from machinelearning.main_speed_detector import SpeedDetector
        frame = np.full((48, 64, 3), 125, np.uint8)
        for cls in (AngleDetector, DistanceDetector, LEDDetector, SpeedDetector):
            with self.subTest(detector=cls.__name__):
                detector = cls.__new__(cls)
                detector.model = Mock()
                detector.model.predict.return_value = [Mock(boxes=None)]
                detector.detect(frame)
                self.assertIs(detector.model.predict.call_args.args[0], frame)
                self.assertTrue((frame == 125).all())

    def test_slow_detections_remain_visible_with_age_and_require_matching_resolution(self):
        from videoprocessing.drone_overlay import draw_recent_drones
        frame = np.zeros((80, 100, 3), np.uint8)
        boxes = [(10, 20, 60, 70, .9)]
        self.assertTrue(draw_recent_drones(frame, boxes, 1, (80, 100), 1.1).any())
        with patch('videoprocessing.drone_overlay.cv2.putText', wraps=cv2.putText) as put_text:
            delayed = draw_recent_drones(frame, boxes, 1, (80, 100), 3)
        np.testing.assert_array_equal(delayed[20, 10], [0, 180, 255])
        self.assertIn('2.0s ago', put_text.call_args_list[0].args[1])
        self.assertIs(draw_recent_drones(frame, boxes, 1, (480, 640), 1.1), frame)
        self.assertIs(draw_recent_drones(frame, [], 1, (80, 100), 3), frame)

    def test_feature_panels_do_not_erase_small_box_outlines(self):
        from videoprocessing.drone_overlay import draw_drones
        frame = np.zeros((200, 300, 3), np.uint8)
        features = [dict(angle=('45', .9), distance=('2m', .8), led=('7', .9), speed=('1m/s', .8))]
        output = draw_drones(frame, [(10, 10, 25, 25, .9)], features)
        for y, x in [(10, 17), (25, 17), (17, 10), (17, 25)]:
            np.testing.assert_array_equal(output[y, x], [0, 220, 0])

    def test_submodel_crops_exclude_overlays_and_clamp_bounds(self):
        vision = ComputerVision()
        frame = np.zeros((80, 100, 3), np.uint8)
        boxes = [Mock(xyxy=[[-10, -10, 60, 60]], conf=[.9]),
                 Mock(xyxy=[[20, 20, 120, 100]], conf=[.8])]
        vision.drone = Mock()
        vision.drone.detect.return_value = Mock(boxes=boxes)
        vision.angle = Mock()
        crops = []
        vision.angle.detect.side_effect = lambda crop: crops.append(crop.copy()) or []
        vision._last_log_t = time.monotonic()
        output = vision.process(frame, dict(drone=True, angle=True, distance=False, led=False, speed=False))
        self.assertEqual([c.shape for c in crops], [(60, 60, 3), (60, 80, 3)])
        self.assertTrue(all(not crop.any() for crop in crops))
        self.assertTrue(output.any())
        self.assertFalse(frame.any())

    def test_drone_detector_keeps_full_frame_and_original_coordinates(self):
        import torch
        from ultralytics.engine.results import Results
        from machinelearning.main_drone_detector import DroneDetector
        for h, w in [(480, 640), (640, 480), (512, 512), (479, 853)]:
            with self.subTest(size=(h, w)):
                detector = DroneDetector.__new__(DroneDetector)
                image = np.zeros((h, w, 3), np.uint8)
                image[:, -20:] = 255  # A target at the edge must reach the model.
                data = torch.tensor([[w-20., 1., w-1., h-1., .9, 0.]])
                result = Results(image, '', {0: 'Drone'}, boxes=data)
                detector.model = Mock()
                detector.model.predict.return_value = [result]
                output, model_input = detector.detect(image, return_preprocessed=True)
                self.assertIs(detector.model.predict.call_args.args[0], image)
                self.assertIs(model_input, image)
                self.assertEqual(detector.model.predict.call_args.kwargs['imgsz'], 640)
                np.testing.assert_allclose(output.boxes.xyxy[0].numpy(), data[0, :4].numpy())
                self.assertEqual(output.orig_shape, (h, w))

    def test_drone_profiles_load_once_and_reuse_their_models(self):
        from videoprocessing.computer_vision import DRONE_MODELS
        vision = ComputerVision()
        indoor, outdoor = Mock(), Mock()
        indoor.detect.return_value = Mock(boxes=[])
        outdoor.detect.return_value = Mock(boxes=[])
        factory = Mock(side_effect=[indoor, outdoor])
        with patch('videoprocessing.computer_vision.import_module', return_value=Mock(DroneDetector=factory)):
            first = vision._detector('drone')
            # The second model is initialized only when the selected profile changes.
            vision.process(np.zeros((80, 100, 3), np.uint8), dict(drone_model='Outdoor'))
            vision.process(np.zeros((80, 100, 3), np.uint8), dict(drone_model='Indoor'))
        self.assertIs(vision.drone, first)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(factory.call_args.kwargs['model_path'], DRONE_MODELS['Outdoor'])

    def test_detection_is_published_before_feature_inference(self):
        vision = ComputerVision()
        vision.drone = Mock()
        vision.drone.detect.return_value = Mock(boxes=[Mock(xyxy=[[10, 10, 60, 60]], conf=[.9])])
        vision._last_log_t = time.monotonic()
        published = []
        def features(crop):
            self.assertEqual(published, [[(10, 10, 60, 60, .9)]])
            return []
        vision.angle = Mock()
        vision.angle.detect.side_effect = features
        vision.process(np.zeros((80, 100, 3), np.uint8),
                       dict(drone=True, angle=True, distance=False, led=False, speed=False),
                       on_detections=published.append)



@unittest.skipUnless(os.environ.get('DISPLAY'), 'Run with xvfb-run for Tk lifecycle checks')
class GuiTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        import posefly_gui
        self.gui = posefly_gui
        self.root = tk.Tk(className='PoseFly')
        self.root.withdraw()
        self.errors = []
        self.root.report_callback_exception = lambda *args: self.errors.append(args)
        self.discovery = patch.object(posefly_gui, 'list_available_cameras', return_value=[])
        self.discovery.start()
        self.app = posefly_gui.PoseFlyGUI(self.root)
        self.temp = tempfile.TemporaryDirectory()
        self.app.out_path_var.set(str(Path(self.temp.name) / 'out.mp4'))
        self.app.save_var.set(False)
        self.app.log_path_var.set(str(Path(self.temp.name) / 'logs' / 'detections.csv'))
        self.pump(lambda: str(self.app.btn_start['state']) == 'normal')

    def tearDown(self):
        self.app._stop_event.set()
        if self.app._thread:
            self.app._thread.join(3)
        if self.app._log_writer:
            self.app._log_writer.stop()
            self.app._log_writer.join(3)
        try:
            for callback in self.root.tk.call('after', 'info'):
                self.root.after_cancel(callback)
            self.root.destroy()
        except self.gui.tk.TclError:
            pass
        self.discovery.stop()
        self.temp.cleanup()
        self.assertEqual(self.errors, [])

    def pump(self, predicate, timeout=3):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.root.update()
            if predicate():
                return
            time.sleep(.005)
        self.fail('GUI condition timed out')

    def test_gui_enables_drone_and_all_features(self):
        self.app._snapshot_settings()
        self.assertEqual(self.app._settings[0], dict(drone=True, angle=True, distance=True, led=True, speed=True, drone_model="Indoor"))
        self.assertFalse(self.app.use_rs_var.get())
        self.assertFalse(hasattr(self.app, 'analysis_preview'))
        self.assertEqual(self.app.feature_table.item('0', 'values'), ('N/A',) * 6)
        self.assertEqual(self.app._log_state, 'off')
        self.assertIsNone(self.app._log_writer)
        self.assertFalse(self.app.backend.cv.log_enabled)

    def test_logging_requires_explicit_opt_in_and_does_not_redirect_stdout(self):
        backend = Mock()
        backend.read_frame.return_value = (True, np.zeros((48, 64, 3), np.uint8))
        backend.process_frame.side_effect = lambda frame, toggles, on_detections=None: frame
        self.app.backend = backend
        stdout = sys.stdout
        self.app.start()
        self.pump(lambda: backend.process_frame.call_count >= 3)
        self.assertIsNone(self.app._log_writer)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertIs(sys.stdout, stdout)
        self.app.stop()
        self.app._thread.join(3)

    def test_logging_can_start_stop_and_append_while_camera_keeps_running(self):
        backend = Mock()
        backend.read_frame.return_value = (True, np.zeros((48, 64, 3), np.uint8))
        backend.process_frame.side_effect = lambda frame, toggles, on_detections=None: frame
        backend.drone_detections.return_value = [(1, 2, 20, 30, .9)]
        backend.drone_features.return_value = [dict(angle=('45', .8), led=(None, None))]
        self.app.backend = backend
        self.app.start()
        self.pump(lambda: backend.process_frame.call_count >= 2)
        self.app.btn_logging.invoke()
        self.pump(lambda: self.app._log_state == 'active' and self.app._log_writer.rows_written > 0)
        self.assertTrue(self.app.log_path_entry.instate(['readonly']))
        self.assertTrue(self.app.btn_log_browse.instate(['disabled']))
        self.app.btn_logging.invoke()
        self.pump(lambda: self.app._log_state == 'stopped')
        with open(self.app.log_path_var.get(), newline='') as file:
            first = list(csv.DictReader(file))
        self.assertEqual(first[0]['angle'], '45')
        self.assertEqual(first[0]['led_id'], 'N/A')
        count = backend.read_frame.call_count
        self.pump(lambda: backend.read_frame.call_count >= count + 3)
        self.app.btn_logging.invoke()
        self.pump(lambda: self.app._log_state == 'active' and self.app._log_writer.rows_written > 0)
        self.app.stop()
        self.pump(lambda: self.app._log_state == 'stopped' and not self.app._thread.is_alive())
        with open(self.app.log_path_var.get(), newline='') as file:
            combined = list(csv.DictReader(file))
        self.assertEqual(combined[:len(first)], first)
        self.assertGreater(len(combined), len(first))
        self.assertNotEqual(first[0]['session_id'], combined[-1]['session_id'])
        backend.open_camera.assert_called_once()
        backend.write_frame_if_enabled.assert_not_called()
        # Restarting capture does not silently re-enable a finished log.
        size = Path(self.app.log_path_var.get()).stat().st_size
        count = backend.process_frame.call_count
        self.app.start()
        self.pump(lambda: backend.process_frame.call_count >= count + 3)
        self.assertEqual(Path(self.app.log_path_var.get()).stat().st_size, size)

    def test_logging_error_does_not_stop_camera(self):
        backend = Mock()
        backend.read_frame.return_value = (True, np.zeros((48, 64, 3), np.uint8))
        backend.process_frame.side_effect = lambda frame, toggles, on_detections=None: frame
        self.app.backend = backend
        parent = Path(self.temp.name) / 'not_a_folder'
        parent.write_text('Keep this file')
        self.app.log_path_var.set(str(parent / 'detections.csv'))
        self.app.start()
        self.app.toggle_logging()
        self.pump(lambda: self.app._log_state == 'error')
        count = backend.read_frame.call_count
        self.pump(lambda: backend.read_frame.call_count >= count + 3)
        self.assertTrue(self.app._running)
        self.assertEqual(parent.read_text(), 'Keep this file')
        self.assertTrue(self.app.btn_logging.instate(['!disabled']))

    def test_about_dialog_can_be_opened_closed_and_reopened(self):
        self.root.deiconify()
        self.root.update()
        self.app.btn_about.invoke()
        self.root.update()
        first = self.app._about
        self.assertTrue(first.winfo_viewable())
        self.assertEqual(first.title(), 'About PoseFly')
        self.assertEqual(str(first.cget('background')), self.gui.BG)
        self.app.btn_about.invoke()
        self.assertIs(self.app._about, first)
        first.destroy()
        self.app.show_about()
        self.root.update()
        self.assertIsNot(self.app._about, first)
        self.assertTrue(self.app._about.winfo_viewable())

    def test_drone_table_keeps_missing_features_and_each_drone_separate(self):
        self.app._detections = [(10, 20, 60, 70, .9), (70, 20, 100, 70, .8)]
        self.app._detections_captured_at = time.monotonic()
        self.app._features = [dict(angle=('45', .8), led=(None, None)), dict(distance=('2m', .7))]
        self.pump(lambda: len(self.app.feature_table.get_children()) == 2)
        self.assertEqual(self.app.feature_table.item('0', 'values'),
                         ('Drone 1', '90%', '45 (80%)', 'N/A', 'N/A', 'N/A'))
        self.assertEqual(self.app.feature_table.item('1', 'values'),
                         ('Drone 2', '80%', 'N/A', '2m (70%)', 'N/A', 'N/A'))
        self.app.tog_drone.set(False)
        self.pump(lambda: self.app.feature_table.item('0', 'values') == ('N/A',) * 6)
        self.assertEqual(len(self.app.feature_table.get_children()), 1)

    def test_preview_resizes_existing_frame_without_cropping(self):
        from PIL import ImageTk
        self.root.geometry('1000x680')
        self.root.deiconify()
        # Distinct corners verify that the whole frame survives resizing.
        frame = np.zeros((360, 640, 3), np.uint8)
        frame[:180, :320] = (0, 0, 255)
        frame[180:, 320:] = (255, 0, 0)
        self.app._show_frame_threadsafe(frame)
        self.pump(lambda: self.app._tk_img is not None and self.app._tk_img.width() > 100)
        first_width = self.app._tk_img.width()
        self.root.geometry('1250x850')
        self.pump(lambda: self.app._tk_img.width() > first_width + 100)
        width, height = self.app._tk_img.width(), self.app._tk_img.height()
        self.assertGreater(width, 640)
        self.assertLessEqual(width, self.app.preview.winfo_width() - 4)
        self.assertLessEqual(height, self.app.preview.winfo_height() - 4)
        self.assertAlmostEqual(width / height, 640 / 360, delta=.01)
        pixels = np.asarray(ImageTk.getimage(self.app._tk_img))
        np.testing.assert_array_equal(pixels[0, 0, :3], [255, 0, 0])
        np.testing.assert_array_equal(pixels[-1, -1, :3], [0, 0, 255])

    def test_focus_preview_keeps_stop_accessible_and_escape_restores_controls(self):
        self.root.geometry('1000x680')
        self.root.deiconify()
        self.root.update()
        original_width = self.app.preview.winfo_width()
        self.app.btn_focus.invoke()
        self.root.update()
        self.assertFalse(self.app.sidebar.winfo_ismapped())
        self.assertGreater(self.app.preview.winfo_width(), original_width + 300)
        self.assertTrue(self.app.btn_stop.winfo_ismapped())
        self.assertTrue(self.app.feature_table.winfo_ismapped())
        self.assertEqual(self.app.btn_focus['text'], 'Show controls')
        self.app.btn_fullscreen.invoke()
        self.assertTrue(self.app._fullscreen)
        self.app._exit_expanded_view()
        self.root.update()
        self.assertFalse(self.app._fullscreen)
        self.assertTrue(self.app.sidebar.winfo_ismapped())
        self.assertEqual(self.app.btn_fullscreen['text'], 'Full screen')
        self.assertEqual(self.app.btn_focus['text'], 'Focus preview')

    def test_session_settings_are_locked_until_camera_stops(self):
        session_controls = (self.app.cam_index_entry, self.app.save_check,
                            self.app.fps_entry, self.app.out_entry, self.app.btn_browse)
        self.app._set_buttons(True)
        for widget in (*session_controls, self.app.cam_dropdown, self.app.btn_start):
            self.assertTrue(widget.instate(['disabled']))
        self.assertTrue(self.app.btn_stop.instate(['!disabled']))
        self.assertTrue(self.app.live_fps_entry.instate(['!disabled']))
        self.app._set_buttons(False)
        for widget in (*session_controls, self.app.btn_start):
            self.assertTrue(widget.instate(['!disabled']))
        self.assertTrue(self.app.cam_dropdown.instate(['readonly', '!disabled']))
        self.assertTrue(self.app.btn_stop.instate(['disabled']))

    def test_live_fps_changes_capture_pacing_without_reopening_or_changing_recording(self):
        backend = Mock()
        reads, owners = [], []
        def read():
            reads.append(time.monotonic())
            owners.append(threading.get_ident())
            return True, np.zeros((48, 64, 3), np.uint8)
        backend.read_frame.side_effect = read
        backend.process_frame.side_effect = lambda frame, toggles, on_detections=None: frame
        backend.drone_detections.return_value = []
        backend.drone_features.return_value = []
        self.app.backend = backend
        self.app.save_var.set(True)
        self.app.fps_var.set('25')
        self.app.live_fps_var.set('1')
        try:
            self.app.start()
            worker = self.app._thread
            self.pump(lambda: backend.write_frame_if_enabled.call_count >= 1)
            self.assertTrue(self.app.live_fps_entry.instate(['!disabled']))
            # Typing applies automatically and interrupts the old one-second wait.
            self.app.live_fps_var.set('60')
            self.pump(lambda: self.app._live_fps == 60)
            marker = len(reads)
            self.pump(lambda: len(reads) >= marker + 5, timeout=.6)
            self.assertLess(float(np.median(np.diff(reads[marker:marker + 5]))), .05)
            # Lowering the target also takes effect in the same capture session.
            self.app.live_fps_var.set('12')
            self.pump(lambda: self.app._live_fps == 12)
            marker = len(reads)
            self.pump(lambda: len(reads) >= marker + 5)
            self.assertGreater(float(np.median(np.diff(reads[marker:marker + 5]))), .06)
            self.assertIs(self.app._thread, worker)
            self.assertEqual(set(owners), {worker.ident})
            backend.open_camera.assert_called_once()
            self.assertTrue(all(call.args[3] == 25. for call in backend.write_frame_if_enabled.call_args_list))
            backend.release_writer.assert_not_called()
            # Stop also wakes the pacing wait instead of waiting for another frame.
            self.app.live_fps_var.set('1')
            self.app.apply_live_fps()
            self.app.stop()
            worker.join(.6)
            self.assertFalse(worker.is_alive())
        finally:
            self.app.stop()
            self.app._thread.join(3)

    def test_invalid_live_fps_keeps_last_valid_target(self):
        self.app.live_fps_var.set('29.97')
        self.assertTrue(self.app.apply_live_fps())
        for value in ('', '0', '-1', '241', 'nan', 'inf', 'invalid'):
            with self.subTest(value=value):
                self.app.live_fps_var.set(value)
                self.assertFalse(self.app.apply_live_fps())
                self.assertEqual(self.app._live_fps, 29.97)
                self.assertEqual(self.app._settings[4], 29.97)
                self.assertIn('Keeping target 29.97', self.app.live_fps_status['text'])

    def test_small_window_keeps_results_visible_and_settings_scrollable(self):
        self.root.geometry('920x600')
        self.root.deiconify()
        self.app.camera_settings_status.configure(text='Camera response. ' * 40)
        self.root.update()
        canvas = self.app._scroll_canvases[0]
        self.assertLess(canvas.yview()[1], 1.)
        self.assertGreater(self.app.feature_table.winfo_height(), 60)
        self.assertLess(self.app.feature_table.winfo_rooty() + self.app.feature_table.winfo_height(),
                        self.app.status.winfo_rooty())
        for button in (self.app.btn_start, self.app.btn_stop, self.app.btn_focus, self.app.btn_fullscreen,
                       self.app.btn_about, self.app.btn_refresh, self.app.btn_apply_rs):
            self.assertTrue(button.winfo_ismapped())
            self.assertGreaterEqual(button.winfo_width(), button.winfo_reqwidth())
            self.assertLessEqual(button.winfo_rootx() + button.winfo_width(), self.root.winfo_rootx() + self.root.winfo_width())
            self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), self.root.winfo_rooty() + self.root.winfo_height())
        # The wheel works over the controls, not only the scrollbar itself.
        self.app._scroll_sidebar(Mock(widget=self.app.iso_entry, num=5, delta=0))
        self.assertGreater(canvas.yview()[0], 0.)
        canvas.yview_moveto(1.)
        self.root.update()
        self.assertAlmostEqual(canvas.yview()[1], 1.)
        self.app.settings_tabs.select(1)
        self.root.update()
        canvas = self.app._scroll_canvases[1]
        for switch in self.app.feature_switches:
            self.assertTrue(switch.winfo_ismapped())
            self.assertLessEqual(switch.winfo_rooty() + switch.winfo_height(), canvas.winfo_rooty() + canvas.winfo_height())
        self.app.settings_tabs.select(2)
        self.root.update()
        self.assertTrue(self.app.btn_browse.winfo_ismapped())
        self.assertGreaterEqual(self.app.btn_browse.winfo_width(), self.app.btn_browse.winfo_reqwidth())
        self.assertLess(self.app.btn_browse.winfo_rooty() + self.app.btn_browse.winfo_height(), self.app.feature_table.winfo_rooty())

    def test_feature_summary_is_visible_then_clears_when_drone_is_lost(self):
        with self.app._frame_lock:
            self.app._detections = [(10, 20, 60, 70, .9)]
            self.app._features = [dict(angle=('45', .8), distance=('2m', .9),
                                       led=('7', .95), speed=('1m/s', .7))]
            self.app._analysis_finished_at = time.monotonic()
            # Even delayed results remain readable, with their source age shown.
            self.app._detections_captured_at = time.monotonic() - 2
        self.pump(lambda: self.app.feature_table.set('0', 'speed') == '1m/s (70%)')
        self.assertIn('source frame', self.app.feature_summary['text'])
        self.app.tog_speed.set(False)
        self.pump(lambda: self.app.feature_table.set('0', 'speed') == 'N/A')
        self.app._detections = []
        self.app._features = []
        self.pump(lambda: 'No drone detected' in self.app.feature_summary['text'])
        self.assertEqual(self.app.feature_table.item('0', 'values'), ('N/A',) * 6)

    def test_delayed_box_renders_without_another_camera_frame_and_clears(self):
        from PIL import ImageTk
        original = np.zeros((80, 100, 3), np.uint8)
        self.app._show_frame_threadsafe(original)
        self.pump(lambda: self.app._tk_img is not None)
        with self.app._frame_lock:
            self.app._detections = [(10, 20, 60, 70, .9)]
            self.app._features = [dict(angle=('45', .8))]
            self.app._detections_captured_at = time.monotonic() - 2
            self.app._detections_shape = (80, 100)
            self.app._overlay_revision += 1
        def has_box():
            pixels = np.asarray(ImageTk.getimage(self.app._tk_img))
            return np.array_equal(pixels[70, 30, :3], [255, 180, 0])
        self.pump(has_box)
        with self.app._frame_lock:
            self.app._detections = []
            self.app._features = []
            self.app._overlay_revision += 1
        self.pump(lambda: not np.asarray(ImageTk.getimage(self.app._tk_img))[:, :, :3].any())

    def test_manual_toggle_is_applied_on_capture_thread(self):
        backend = Mock()
        backend.read_frame.return_value = (True, np.full((48, 64, 3), 125, np.uint8))
        backend.process_frame.side_effect = lambda frame, toggles, on_detections=None: frame
        backend.drone_detections.return_value = []
        backend.set_rollingshutter.return_value = 'ISO control unsupported'
        self.app.use_rs_var.set(True)
        self.app.backend = backend
        self.app.start()
        self.pump(lambda: 'ISO control unsupported' in self.app.camera_settings_status['text'])
        previous_calls = backend.set_rollingshutter.call_count
        self.app.apply_rs()
        self.pump(lambda: backend.set_rollingshutter.call_count > previous_calls)
        self.app.use_rs_var.set(False)
        self.pump(lambda: backend.set_rollingshutter.call_args.kwargs.get('enabled') is False)
        self.app.stop()
        self.app._thread.join(3)

    def test_manual_camera_index_is_respected(self):
        self.app._populate_cameras([{'index': 2, 'label': 'Camera 2'}])
        self.app.cam_index_var.set(42)
        self.assertEqual(self.app._selected_camera_index(), 42)

    def test_worker_error_cleans_up_and_remains_visible(self):
        backend = Mock()
        backend.apply_camera_settings_led_id.side_effect = RuntimeError('settings failure')
        self.app.backend = backend
        with patch.object(self.gui.traceback, 'print_exc'):
            self.app.start()
            self.pump(lambda: 'ERROR' in self.app.status['text'])
        backend.release_camera.assert_called_once()
        backend.release_writer.assert_called_once()
        self.assertIn('settings failure', self.app.status['text'])
        self.assertEqual(str(self.app.btn_start['state']), 'normal')

    def test_close_waits_for_worker_without_cross_thread_tk_or_release(self):
        backend = Mock()
        entered, release = threading.Event(), threading.Event()
        owner_threads = []
        def read():
            entered.set()
            release.wait(2)
            return True, np.zeros((48, 64, 3), np.uint8)
        backend.read_frame.side_effect = read
        backend.process_frame.side_effect = lambda frame, toggles, on_detections=None: frame
        backend.release_camera.side_effect = lambda: owner_threads.append(threading.get_ident())
        backend.release_writer.side_effect = lambda: owner_threads.append(threading.get_ident())
        self.app.backend = backend
        self.app.start()
        self.pump(entered.is_set)
        worker = self.app._thread
        self.app.stop()
        self.app.start()
        self.assertIs(self.app._thread, worker)
        self.app.on_close()
        backend.release_camera.assert_not_called()
        release.set()
        worker.join(3)  # Worker must finish even while Tk's event loop is idle.
        self.assertFalse(worker.is_alive())
        self.assertEqual(owner_threads, [worker.ident, worker.ident])

    def test_live_preview_continues_during_slow_analysis_and_stop_waits(self):
        backend = Mock()
        entered, release = threading.Event(), threading.Event()
        captured = []
        def read():
            index = len(captured) + 1
            captured.append(index)
            return True, np.full((48, 64, 3), index % 255, np.uint8)
        def analyze(frame, toggles, on_detections=None):
            entered.set()
            release.wait(3)
            frame[:] = 255  # Inference annotations must not change live camera frames.
            return frame
        backend.read_frame.side_effect = read
        backend.process_frame.side_effect = analyze
        self.app.backend = backend
        self.app.use_rs_var.set(False)
        shown = []
        original_show = self.app._show_frame_threadsafe
        def show(frame):
            shown.append(frame)
            original_show(frame)
        self.app._show_frame_threadsafe = show
        try:
            self.app.start()
            self.pump(entered.is_set)
            self.pump(lambda: len(shown) >= 12)
            self.assertEqual(backend.process_frame.call_count, 1)
            self.assertGreater(int(shown[-1][0, 0, 0]), int(shown[0][0, 0, 0]))
            self.app.stop()
            self.assertTrue(self.app._thread.is_alive())
            backend.release_writer.assert_not_called()
        finally:
            self.app._stop_event.set()
            release.set()
            self.app._thread.join(3)
        self.assertFalse(self.app._thread.is_alive())
        self.assertEqual(int(shown[0][0, 0, 0]), 1)
        backend.release_writer.assert_called_once()
        backend.release_camera.assert_called_once()

    def test_drone_is_shown_while_features_are_still_running(self):
        backend = Mock()
        release = threading.Event()
        detections = [(10, 10, 60, 60, .9)]
        backend.read_frame.return_value = (True, np.zeros((80, 100, 3), np.uint8))
        backend.drone_detections.return_value = detections
        backend.drone_features.return_value = [dict(angle=('45', .8))]
        def analyze(frame, toggles, on_detections=None):
            on_detections(detections)
            release.wait(3)
            return frame
        backend.process_frame.side_effect = analyze
        self.app.backend = backend
        try:
            self.app.start()
            self.pump(lambda: self.app.feature_table.set('0', 'drone') == 'Drone 1')
            self.assertEqual(self.app.feature_table.item('0', 'values')[2:], ('N/A',) * 4)
            self.assertIn('estimating selected features', self.app.feature_summary['text'])
            self.assertIsNone(self.app._analysis_finished_at)
            self.assertEqual(self.app._detections, detections)
        finally:
            self.app.stop()
            release.set()
            self.app._thread.join(3)

    def test_analysis_failure_stops_capture_and_reports_error(self):
        backend = Mock()
        backend.read_frame.return_value = (True, np.zeros((48, 64, 3), np.uint8))
        backend.process_frame.side_effect = RuntimeError('analysis failure')
        self.app.backend = backend
        with patch.object(self.gui.traceback, 'print_exc'):
            self.app.start()
            self.pump(lambda: 'analysis failure' in self.app.status['text'])
        self.assertFalse(self.app._thread.is_alive())
        backend.release_writer.assert_called_once()
        backend.release_camera.assert_called_once()


if __name__ == '__main__':
    unittest.main()
