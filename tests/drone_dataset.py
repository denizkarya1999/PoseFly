"""Load labelled drone test images without extracting the dataset archive."""
from pathlib import Path
import zipfile
import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / 'Machine Learning/Datasets/Posefly_Drone.v2i.yolov11.zip'


def samples():
    with zipfile.ZipFile(DATASET) as archive:
        names = sorted(name for name in archive.namelist()
                       if name.startswith('test/images/') and name.lower().endswith(('.jpg', '.png')))
        for name in names:
            frame = cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f'Unreadable dataset image: {name}')
            label = name.replace('/images/', '/labels/').rsplit('.', 1)[0] + '.txt'
            boxes = []
            for line in archive.read(label).decode().splitlines():
                _, cx, cy, w, h = map(float, line.split())
                boxes.append(np.array([cx-w/2, cy-h/2, cx+w/2, cy+h/2]) *
                             [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            yield name, frame, boxes


def iou(a, b):
    dimensions = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection = dimensions.prod()
    union = (a[2:]-a[:2]).prod() + (b[2:]-b[:2]).prod() - intersection
    return float(intersection / max(1, union))
