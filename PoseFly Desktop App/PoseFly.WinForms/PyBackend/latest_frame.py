"""A one-frame mailbox: slow consumers always receive the newest camera frame."""
import threading


class LatestFrame:
    def __init__(self):
        self._condition = threading.Condition()
        self._item = None
        self._closed = False

    def put(self, item):
        with self._condition:
            if not self._closed:
                self._item = item
                self._condition.notify()

    def get(self):
        with self._condition:
            self._condition.wait_for(lambda: self._closed or self._item is not None)
            if self._closed:
                return None
            item, self._item = self._item, None
            return item

    def close(self):
        with self._condition:
            self._closed = True
            self._item = None
            self._condition.notify_all()
