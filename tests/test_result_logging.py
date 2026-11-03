"""CSV logging opt-in lifecycle, data integrity, and storage-failure coverage."""
import csv
from datetime import datetime, timezone
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'PoseFly Desktop App/PoseFly.WinForms/PyBackend'))
from result_logging import FIELDS, ResultLogWriter

CAPTURED = datetime(2026, 11, 2, 20, 15, 30, 123000, tzinfo=timezone.utc)


class ResultLoggingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'results' / 'detections.csv'
        self.writers = []

    def tearDown(self):
        for writer in self.writers:
            writer.stop()
            writer.join(3)
            self.assertFalse(writer.is_alive())
        self.directory.cleanup()

    def writer(self, **kwargs):
        events = []
        writer = ResultLogWriter(self.path, lambda *event: events.append(event), **kwargs)
        self.writers.append(writer)
        return writer, events

    def finish(self, writer):
        writer.stop()
        writer.join(3)
        self.assertFalse(writer.is_alive())

    def rows(self):
        with self.path.open(newline='') as file:
            return list(csv.DictReader(file))

    def test_timestamp_boxes_features_and_missing_values_for_each_drone(self):
        writer, events = self.writer()
        writer.start()
        writer.submit([(1, 2, 20, 30, .9), (5, 6, 50, 60, .8)],
                      [dict(angle=('45', .7), led=(None, None)), dict(distance=('2m', .6), speed=('slow, stable', .5))],
                      CAPTURED, 'Indoor', True)
        writer.submit([], [], CAPTURED, 'Outdoor', False)
        self.finish(writer)
        rows = self.rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]['captured_at_utc'], '2026-11-02T20:15:30.123+00:00')
        self.assertEqual(rows[0]['confidence'], '0.9')
        self.assertEqual(rows[0]['angle'], '45')
        self.assertEqual(rows[0]['led_id'], 'N/A')
        self.assertEqual(rows[0]['distance_confidence'], 'N/A')
        self.assertEqual(rows[1]['x1'], '5')
        self.assertEqual(rows[1]['distance'], '2m')
        self.assertEqual(rows[1]['speed'], 'slow, stable')
        self.assertEqual(rows[0]['frame'], rows[1]['frame'])
        self.assertEqual(rows[2]['frame'], '2')
        self.assertEqual(rows[2]['drone_count'], '0')
        self.assertEqual(rows[2]['detection_enabled'], 'False')
        for key in ('drone', 'confidence', 'x1', 'angle', 'distance', 'led_id', 'speed'):
            self.assertEqual(rows[2][key], 'N/A')
        self.assertEqual(events[-1], ('stopped', 3, ''))
        self.assertFalse(writer.submit([], [], CAPTURED, 'Indoor', True))

    def test_restart_appends_without_duplicate_header_or_losing_rows(self):
        first, _ = self.writer()
        first.start()
        first.submit([], [], CAPTURED, 'Indoor', True)
        self.finish(first)
        original = self.path.read_bytes()
        second, _ = self.writer()
        second.start()
        second.submit([], [], CAPTURED, 'Outdoor', True)
        self.finish(second)
        self.assertTrue(self.path.read_bytes().startswith(original))
        rows = self.rows()
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]['session_id'], rows[1]['session_id'])

    def test_compatible_file_without_final_newline_remains_valid(self):
        self.path.parent.mkdir()
        self.path.write_text(','.join(FIELDS))
        writer, _ = self.writer()
        writer.start()
        writer.submit([], [], CAPTURED, 'Indoor', True)
        self.finish(writer)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(self.rows()[0]['drone_model'], 'Indoor')

    def test_incompatible_existing_file_is_preserved(self):
        self.path.parent.mkdir()
        self.path.write_text('Keep this unrelated file\n')
        writer, events = self.writer()
        writer.start()
        writer.join(3)
        self.assertEqual(events[-1][0], 'error')
        self.assertEqual(self.path.read_text(), 'Keep this unrelated file\n')
        self.assertFalse(writer.submit([], [], CAPTURED, 'Indoor', True))

    def test_write_failure_is_reported_and_closes_the_file(self):
        class FullDisk(io.StringIO):
            def write(self, text):
                if self.tell():
                    raise OSError('disk full')
                return super().write(text)
        file = FullDisk()
        writer, events = self.writer()
        with patch.object(Path, 'open', return_value=file):
            writer.start()
            writer.submit([], [], CAPTURED, 'Indoor', True)
            self.finish(writer)
        self.assertTrue(file.closed)
        self.assertEqual(events[-1], ('error', 0, 'disk full'))

    def test_backlog_is_bounded_and_reports_incomplete_logging(self):
        writer, events = self.writer(max_pending=1)
        self.assertTrue(writer.submit([], [], CAPTURED, 'Indoor', True))
        self.assertFalse(writer.submit([], [], CAPTURED, 'Indoor', True))
        writer.start()
        writer.join(3)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual(events[-1][0], 'error')
        self.assertIn('could not keep up', events[-1][2])
