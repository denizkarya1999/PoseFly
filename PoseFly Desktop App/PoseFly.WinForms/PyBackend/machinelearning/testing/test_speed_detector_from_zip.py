import os
import re
import zipfile
import tempfile
import tkinter as tk
from tkinter import filedialog
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 🔹 IMPORT THE CLASS YOU PROVIDED (no redefinition)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from main_speed_detector import SpeedDetector

# -------------------- CONFIG --------------------
CLASSES = ["fast", "slow", "static"]
IDX = {c: i for i, c in enumerate(CLASSES)}


# -------------------- UI --------------------
def pick_zip_file():
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select ZIP with speed images",
        filetypes=[("ZIP files", "*.zip")],
    )
    root.destroy()
    return path


# -------------------- Ground truth from filename --------------------
def gt_from_filename(path: str):
    """
    Accepts filenames like:
      Fast_xxx.jpg
      PoseFly_Fast_xxx.jpg
      something-Slow-xxx.png
      test_Static_frame.jpeg
    """

    name = os.path.basename(path)

    m = re.search(r"(Fast|Slow|Static)", name, flags=re.IGNORECASE)

    return m.group(1).lower() if m else None


# -------------------- Confusion matrix plot (🍊 ORANGE THEME) --------------------
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(6.5, 5.5))

    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Oranges)
    plt.title("Confusion Matrix (fast / slow / static)")
    plt.colorbar(im, label="Count")

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)

    thresh = cm.max() * 0.6 if cm.max() > 0 else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
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


# -------------------- MAIN TEST --------------------
def main():
    zip_path = pick_zip_file()
    if not zip_path:
        print("No ZIP selected.")
        return

    detector = SpeedDetector()
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)

    total = used = skipped_gt = skipped_read = skipped_pred = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_path, "r") as z:
            for name in z.namelist():
                if name.endswith("/"):
                    continue

                total += 1
                gt = gt_from_filename(name)
                if gt is None:
                    skipped_gt += 1
                    continue

                img_path = os.path.join(tmpdir, os.path.basename(name))
                with z.open(name) as src, open(img_path, "wb") as dst:
                    dst.write(src.read())

                img = cv2.imread(img_path)
                if img is None:
                    skipped_read += 1
                    continue

                out = detector.detect(img)
                if not out:
                    skipped_pred += 1
                    continue

                _, pred_label, _ = out[0]
                pred = pred_label.lower()

                if pred not in IDX:
                    skipped_pred += 1
                    continue

                cm[IDX[gt], IDX[pred]] += 1
                used += 1

    print("ZIP:", zip_path)
    print("Total images:", total)
    print("Used in CM:", used)
    print("Skipped (no GT):", skipped_gt)
    print("Skipped (read fail):", skipped_read)
    print("Skipped (no prediction / unknown label):", skipped_pred)
    print("\nConfusion Matrix (rows=GT, cols=Pred) order:", CLASSES)
    print(cm)

    plot_confusion_matrix(cm, CLASSES)


if __name__ == "__main__":
    main()