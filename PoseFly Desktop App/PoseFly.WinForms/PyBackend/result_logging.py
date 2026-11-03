"""Optional CSV result logging, with file I/O isolated from capture and Tk."""
import csv
from datetime import datetime, timezone
from pathlib import Path
import queue
import threading
import time
import uuid


FIELDS = (
    'session_id', 'frame', 'captured_at_utc', 'logged_at_utc', 'drone_model',
    'detection_enabled', 'drone_count', 'drone', 'confidence', 'x1', 'y1', 'x2', 'y2',
    'angle', 'angle_confidence', 'distance', 'distance_confidence',
    'led_id', 'led_id_confidence', 'speed', 'speed_confidence',
)


class ResultLogWriter:
    def __init__(self, path, notify, max_pending=256):
        self.path = Path(path).expanduser()
        self.session_id = uuid.uuid4().hex
        self._notify = notify
        self._max_pending = max_pending
        self._queue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._accepting = True
        self._failure = None
        self._frame = 0
        self.rows_written = 0
        self._thread = threading.Thread(target=self._run, name='PoseFly result log', daemon=True)

    def start(self):
        self._thread.start()

    def is_alive(self):
        return self._thread.is_alive()

    def join(self, timeout=None):
        self._thread.join(timeout)

    def stop(self):
        with self._lock:
            if self._accepting:
                self._accepting = False
                self._queue.put(None)

    def submit(self, detections, features, captured_at, model, detection_enabled):
        with self._lock:
            if not self._accepting:
                return False
            if self._queue.qsize() >= self._max_pending:
                self._failure = 'Log storage could not keep up. Logging stopped; choose faster storage.'
                self._accepting = False
                self._queue.put(None)
                return False
            self._frame += 1
            common = dict(session_id=self.session_id, frame=self._frame,
                          captured_at_utc=captured_at.isoformat(timespec='milliseconds'),
                          logged_at_utc=datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
                          drone_model=model, detection_enabled=bool(detection_enabled),
                          drone_count=len(detections))
            rows = []
            for index in range(max(1, len(detections))):
                row = dict.fromkeys(FIELDS, 'N/A')
                row.update(common)
                if detections:
                    x1, y1, x2, y2, confidence = detections[index]
                    row.update(drone=index + 1, confidence=confidence, x1=x1, y1=y1, x2=x2, y2=y2)
                    estimates = features[index] if index < len(features) else {}
                    for name, column in (('angle', 'angle'), ('distance', 'distance'), ('led', 'led_id'), ('speed', 'speed')):
                        label, confidence = estimates.get(name, (None, None))
                        if label is not None and confidence is not None:
                            row[column], row[column + '_confidence'] = label, confidence
                rows.append(row)
            self._queue.put(rows)
            return True

    def _run(self):
        file = None
        failure = None
        try:
            if self.path.suffix.lower() != '.csv':
                raise ValueError('Choose a .csv file for the detection log.')
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Append only to a compatible log; never truncate an existing file.
            file = self.path.open('a+', encoding='utf-8', newline='')
            file.seek(0)
            header = next(csv.reader(file), None)
            if header is not None and header != list(FIELDS):
                raise ValueError('This file is not a PoseFly CSV log. Choose a different file.')
            file.seek(0, 2)
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            if header is None:
                writer.writeheader()
                file.flush()
            else:
                with self.path.open('rb') as tail:
                    tail.seek(-1, 2)
                    needs_newline = tail.read(1) not in (b'\n', b'\r')
                if needs_newline:
                    file.write('\n')
                    file.flush()
            self._notify('active', 0, '')
            last_update = time.monotonic()
            while True:
                rows = self._queue.get()
                if rows is None:
                    break
                writer.writerows(rows)
                file.flush()
                self.rows_written += len(rows)
                now = time.monotonic()
                if now - last_update >= .2:
                    self._notify('active', self.rows_written, '')
                    last_update = now
        except Exception as exc:
            failure = str(exc)
        finally:
            if file is not None:
                try:
                    file.close()
                except OSError as exc:
                    failure = failure or str(exc)
            with self._lock:
                self._accepting = False
                failure = failure or self._failure
            self._notify('error' if failure else 'stopped', self.rows_written, failure or '')
