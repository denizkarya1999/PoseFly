"""Camera discovery shared by the Linux GUI and standalone recorder."""
import os
import platform

import cv2


def _linux_video_indices(max_index=None):
    # Sparse device numbers are normal (USB cameras may also expose metadata nodes).
    try:
        return sorted({int(name[5:]) for name in os.listdir('/dev')
                       if name.startswith('video') and name[5:].isdigit()
                       and (max_index is None or int(name[5:]) <= max_index)})
    except OSError:
        return []


def _probe_camera(index, backend):
    cap = None
    try:
        cap = cv2.VideoCapture(int(index), backend)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                return True, (frame.shape[1], frame.shape[0])
        return False, None
    except cv2.error:
        return False, None
    finally:
        if cap is not None:
            cap.release()


def list_available_cameras(max_index=None):
    system = platform.system()
    if system == 'Linux':
        indices = _linux_video_indices(max_index)
        backends = [(cv2.CAP_V4L2, 'V4L2')]
    else:
        indices = range((20 if max_index is None else max_index) + 1)
        backends = ([(cv2.CAP_DSHOW, 'DShow'), (cv2.CAP_MSMF, 'MSMF'), (cv2.CAP_ANY, 'Any')]
                    if system == 'Windows' else [(cv2.CAP_ANY, 'Any')])
    found = []
    for index in indices:
        for backend, name in backends:
            ok, size = _probe_camera(index, backend)
            if ok:
                found.append({'index': index,
                              'label': f'Camera {index}  ({name}, {size[0]}x{size[1]})'})
                break
    return found
