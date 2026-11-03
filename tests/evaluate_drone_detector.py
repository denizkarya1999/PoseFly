"""Evaluate a checkpoint on the 40 labelled test images; no webcam is accessed."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'PoseFly Desktop App/PoseFly.WinForms/PyBackend'))
from drone_dataset import samples, iou
from machinelearning.main_drone_detector import DroneDetector, DEFAULT_MODEL
from videoprocessing.computer_vision import DRONE_MODELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Indoor', help='Indoor, Outdoor, or checkpoint path')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    path = DRONE_MODELS.get(args.model, args.model)
    detector = DroneDetector(path)
    detector.model.overrides['device'] = args.device
    images = total = matched = predictions = detected_images = 0
    overlaps = []
    for _, frame, truth in samples():
        result = detector.detect(frame)
        pred = result.boxes.xyxy.cpu().numpy()
        images += 1
        total += len(truth)
        predictions += len(pred)
        detected_images += bool(len(pred))
        candidates = sorted(((iou(gt, p), g, k) for g, gt in enumerate(truth)
                             for k, p in enumerate(pred)), reverse=True)
        used_truth, used_pred = set(), set()
        for overlap, g, k in candidates:
            if overlap >= .5 and g not in used_truth and k not in used_pred:
                used_truth.add(g)
                used_pred.add(k)
                matched += 1
                overlaps.append(overlap)
    report = dict(checkpoint=path, confidence_threshold=.1, matching_iou=.5, images=images,
                  images_with_predictions=detected_images, ground_truth_boxes=total,
                  matched_boxes=matched, predicted_boxes=predictions,
                  mean_matched_iou=float(np.mean(overlaps)) if overlaps else 0)
    output = json.dumps(report, indent=2) + '\n'
    print(output, end='')
    if args.output:
        args.output.write_text(output)


if __name__ == '__main__':
    main()
