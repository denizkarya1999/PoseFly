import os
import zipfile
import tempfile
import tkinter as tk
from tkinter import filedialog
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 🔹 IMPORT YOUR DETECTOR CLASS
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main_drone_detector import DroneDetector


# -------------------- CONFIG --------------------
CLASSES = ["drone", "not_drone"]  # not_drone == "missed"
IDX = {c: i for i, c in enumerate(CLASSES)}
SPLIT_ORDER = ["training", "validation", "testing"]


# -------------------- UI (MULTI ZIP PICKER) --------------------
def pick_zip_files():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    initialdir = "/mnt/data" if os.path.isdir("/mnt/data") else os.getcwd()

    paths = filedialog.askopenfilenames(
        title="Select PoseFly Drone ZIP files (Training / Validation / Testing)",
        initialdir=initialdir,
        filetypes=[("ZIP files", "*.zip")],
    )

    root.destroy()
    return list(paths)


# -------------------- Ground Truth --------------------
def gt_label_for_any_image(_path: str):
    return "drone"  # Drone-only dataset


# -------------------- Helper: unwrap (result, preprocessed) --------------------
def unwrap_detection(detect_return):
    """
    Supports DroneDetector.detect returning either:
      - result
      - (result, preprocessed_img)
    """
    if isinstance(detect_return, tuple) and len(detect_return) == 2:
        return detect_return[0], detect_return[1]
    return detect_return, None


# -------------------- Ultralytics Results helpers --------------------
def pred_from_detector_output(det_out):
    """
    Requirement:
      If it ever generates a bounding box -> detected.

    Works with Ultralytics Results object:
      det_out.boxes is a Boxes object and len(det_out.boxes) is #detections.
    """
    if det_out is None:
        return "not_drone"

    boxes = getattr(det_out, "boxes", None)
    if boxes is None:
        return "not_drone"

    return "drone" if len(boxes) > 0 else "not_drone"


def best_detection(det_out):
    """
    For visualization: return best (label, xyxy, conf) if any box exists.
    Label is forced to 'drone' since dataset is single-class.
    """
    if det_out is None:
        return None

    boxes = getattr(det_out, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return None

    confs = boxes.conf
    xyxys = boxes.xyxy

    confs_np = confs.detach().cpu().numpy()
    i = int(np.argmax(confs_np))
    xyxy = xyxys[i].detach().cpu().numpy().tolist()  # [x1,y1,x2,y2]
    conf = float(confs_np[i])

    return ("drone", xyxy, conf)


def draw_xyxy(vis_bgr, xyxy, label="drone", conf=None):
    h, w = vis_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, xyxy[:4])

    x1 = int(max(0, min(w - 1, round(x1))))
    y1 = int(max(0, min(h - 1, round(y1))))
    x2 = int(max(0, min(w - 1, round(x2))))
    y2 = int(max(0, min(h - 1, round(y2))))

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    cv2.rectangle(vis_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)

    txt = label
    if conf is not None:
        txt += f" ({conf:.3f})"

    cv2.putText(
        vis_bgr,
        txt,
        (x1, max(20, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


# -------------------- Confusion matrix plot --------------------
def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6.5, 5.5))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Oranges)
    plt.title(title)
    plt.colorbar(im, label="Count")

    ticks = np.arange(len(CLASSES))
    plt.xticks(ticks, CLASSES)
    plt.yticks(ticks, CLASSES)

    thresh = cm.max() * 0.6 if cm.max() > 0 else 0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.show()


def print_recall_from_cm(cm):
    tp = int(cm[IDX["drone"], IDX["drone"]])
    fn = int(cm[IDX["drone"], IDX["not_drone"]])
    denom = tp + fn
    recall = (tp / denom) if denom > 0 else 0.0
    print(f"TP: {tp} | FN (missed): {fn} | Recall: {recall:.4f}")


# -------------------- ZIP PROCESSING (CM) --------------------
def process_zip(zip_path, detector):
    cm = np.zeros((2, 2), dtype=np.int64)
    total = used = skipped_read = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                if name.endswith("/"):
                    continue

                total += 1
                gt = gt_label_for_any_image(name)

                img_path = os.path.join(tmpdir, os.path.basename(name))
                with z.open(name) as src, open(img_path, "wb") as dst:
                    dst.write(src.read())

                img = cv2.imread(img_path)
                if img is None:
                    skipped_read += 1
                    continue

                det_ret = detector.detect(img, return_preprocessed=True)
                det_out, _pre = unwrap_detection(det_ret)

                pred = pred_from_detector_output(det_out)
                cm[IDX[gt], IDX[pred]] += 1
                used += 1

    return cm, total, used, skipped_read


# -------------------- SAMPLE IMAGE (ORIGINAL + PREPROCESSED) --------------------
def show_one_sample_with_box(zip_path, detector, split_name):
    """
    Shows:
      - Original image
      - Preprocessed image (returned by DroneDetector.detect)
      - Draws bbox on the preprocessed image (boxes are in that coordinate space)
    """
    chosen_img = None
    chosen_name = None
    chosen_det = None  # (label, xyxy, conf)
    chosen_pre = None  # preprocessed image

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = [n for n in z.namelist() if not n.endswith("/")]

            # Pass 1: prefer an image WITH ANY detection
            for name in names:
                img_path = os.path.join(tmpdir, os.path.basename(name))
                with z.open(name) as src, open(img_path, "wb") as dst:
                    dst.write(src.read())

                img = cv2.imread(img_path)
                if img is None:
                    continue

                det_ret = detector.detect(img, return_preprocessed=True)
                det_out, pre_img = unwrap_detection(det_ret)

                det = best_detection(det_out)
                if det is not None:
                    chosen_img, chosen_name, chosen_det, chosen_pre = img, name, det, pre_img
                    break

            # Pass 2: fallback to first readable image (even if no detection)
            if chosen_img is None:
                for name in names:
                    img_path = os.path.join(tmpdir, os.path.basename(name))
                    with z.open(name) as src, open(img_path, "wb") as dst:
                        dst.write(src.read())

                    img = cv2.imread(img_path)
                    if img is not None:
                        det_ret = detector.detect(img, return_preprocessed=True)
                        det_out, pre_img = unwrap_detection(det_ret)
                        det = best_detection(det_out)

                        chosen_img, chosen_name, chosen_pre, chosen_det = img, name, pre_img, det
                        break

    if chosen_img is None:
        print(f"[{split_name}] Could not read any images from: {zip_path}")
        return

    # Original view (no bbox here because boxes are in preprocessed space)
    original_vis = chosen_img.copy()

    # Preprocessed view (what model sees)
    pre_vis = chosen_pre.copy() if chosen_pre is not None else chosen_img.copy()

    title_extra = " — NO DETECTION"
    if chosen_det is not None:
        label, xyxy, conf = chosen_det
        draw_xyxy(pre_vis, xyxy, label=label, conf=conf)
        title_extra = " — DETECTED"

    original_rgb = cv2.cvtColor(original_vis, cv2.COLOR_BGR2RGB)
    pre_rgb = cv2.cvtColor(pre_vis, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.axis("off")
    plt.title(f"{split_name.capitalize()} ORIGINAL\n{os.path.basename(chosen_name)}")

    plt.subplot(1, 2, 2)
    plt.imshow(pre_rgb)
    plt.axis("off")
    plt.title(f"{split_name.capitalize()} PREPROCESSED{title_extra}\n(resize_fill_512 from DroneDetector)")

    plt.tight_layout()
    plt.show()


def split_name_from_zip(zip_path: str):
    b = os.path.basename(zip_path).lower()
    if "train" in b:
        return "training"
    if "val" in b:
        return "validation"
    if "test" in b:
        return "testing"
    return "dataset"


# -------------------- MAIN --------------------
def main():
    zip_paths = pick_zip_files()
    if not zip_paths:
        print("No ZIP files selected.")
        return

    detector = DroneDetector()
    overall_cm = np.zeros((2, 2), dtype=np.int64)

    def sort_key(zp):
        s = split_name_from_zip(zp)
        return SPLIT_ORDER.index(s) if s in SPLIT_ORDER else 999

    zip_paths = sorted(zip_paths, key=sort_key)

    for zp in zip_paths:
        split_name = split_name_from_zip(zp)

        cm, total, used, skipped_read = process_zip(zp, detector)
        overall_cm += cm

        print("\nZIP:", zp)
        print("Split:", split_name)
        print("Total images:", total)
        print("Used in CM:", used)
        print("Skipped (read fail):", skipped_read)
        print("CM (rows=GT, cols=Pred) order:", CLASSES)
        print(cm)
        print_recall_from_cm(cm)

        plot_confusion_matrix(cm, f"Confusion Matrix — {os.path.basename(zp)}")

        # Show ORIGINAL + PREPROCESSED + bbox
        show_one_sample_with_box(zp, detector, split_name)

    print("\n=== OVERALL (ALL SELECTED ZIPS) ===")
    print(overall_cm)
    print_recall_from_cm(overall_cm)
    plot_confusion_matrix(overall_cm, "Confusion Matrix — Overall (All Selected ZIPs)")


if __name__ == "__main__":
    main()