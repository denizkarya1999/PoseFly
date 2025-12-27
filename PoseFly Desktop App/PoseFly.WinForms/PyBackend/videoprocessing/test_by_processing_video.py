# run_video_file_save_only.py
"""
Select a video file, process all frames using ComputerVision.process(),
save the processed output video, show a Tkinter progress bar,
and LOG RESULTS ONLY (the iteration lines printed by ComputerVision) to a .txt file.

- No OpenCV preview window
- Progress window stays responsive
- If frame count is unknown, the bar switches to "indeterminate"
- Results-only log file: <input>_results_only.txt
"""

import os
import sys
import time
import cv2
import tkinter as tk
from tkinter import filedialog, ttk

# ---- Make sure PyBackend root is on sys.path so `machinelearning` imports work ----
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from computer_vision import ComputerVision


# ---------------------------
# Results-only logging: capture ONLY the iteration lines (and nothing else)
# ---------------------------
class ResultsOnlyLogger:
    """
    Redirects sys.stdout to a file but only writes lines that look like:
      Iteration-123: ...
    Everything else is dropped (so your console prints don't pollute the log).
    """

    def __init__(self, file_path: str, keep_console: bool = True):
        self.file_path = file_path
        self.file = open(file_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self.keep_console = keep_console
        self._buf = ""

    def write(self, s: str):
        # Optional: still show normal prints in console
        if self.keep_console:
            self._stdout.write(s)

        # Buffer until newline so we can filter by line
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            line = line.rstrip("\r")
            if line.startswith("Iteration-"):
                self.file.write(line + "\n")
                self.file.flush()

    def flush(self):
        if self.keep_console:
            self._stdout.flush()
        self.file.flush()

    def close(self):
        # flush remaining buffer as a line (rare)
        if self._buf.strip().startswith("Iteration-"):
            self.file.write(self._buf.strip() + "\n")
        self.file.flush()
        self.file.close()


def pick_video_file() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[
            ("Video files", "*.mp4 *.mkv *.avi *.mov *.m4v *.wmv *.webm"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return path or ""


def clamp_fps(fps: float, fallback: float = 30.0) -> float:
    if fps is None or fps != fps or fps <= 1.0 or fps > 240.0:
        return fallback
    return float(fps)


def make_output_path(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return base + "_processed.mp4"


def make_results_log_path(input_path: str) -> str:
    base, _ = os.path.splitext(input_path)
    return base + "_results_only.txt"


def try_create_writer(out_path: str, fps: float, w: int, h: int) -> tuple[cv2.VideoWriter, str]:
    """
    Try a few common codecs. If one fails, fall back to .avi + MJPG.
    Returns (writer, actual_output_path).
    """
    candidates = [
        (out_path, "mp4v"),  # usually available on Windows
        (out_path, "avc1"),
        (out_path, "H264"),
    ]

    for path, fourcc_str in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if vw.isOpened():
            print(f"Writer opened: {path}  codec={fourcc_str}  fps={fps:.2f}  size={w}x{h}")
            return vw, path

    avi_path = os.path.splitext(out_path)[0] + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
    if vw.isOpened():
        print(f"Writer fallback opened: {avi_path}  codec=MJPG  fps={fps:.2f}  size={w}x{h}")
        return vw, avi_path

    raise RuntimeError("Failed to open VideoWriter with all attempted codecs.")


class ProgressUI:
    def __init__(self, total_frames: int):
        self.root = tk.Tk()
        self.root.title("PoseFly - Processing Video")
        self.root.resizable(False, False)

        self.total_frames = total_frames if total_frames and total_frames > 0 else 0

        pad = 12
        frm = ttk.Frame(self.root, padding=pad)
        frm.grid(row=0, column=0, sticky="nsew")

        self.label = ttk.Label(frm, text="Starting...")
        self.label.grid(row=0, column=0, sticky="w")

        mode = "determinate" if self.total_frames > 0 else "indeterminate"
        self.bar = ttk.Progressbar(frm, length=420, mode=mode, maximum=max(1, self.total_frames))
        self.bar.grid(row=1, column=0, pady=(8, 0), sticky="ew")

        self.detail = ttk.Label(frm, text="")
        self.detail.grid(row=2, column=0, pady=(8, 0), sticky="w")

        if mode == "indeterminate":
            self.bar.start(10)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._closed = False

        self._last_ui_t = 0.0
        self._ui_every_sec = 0.10

    def _on_close(self):
        self._closed = True
        self.root.withdraw()

    def update(self, frame_idx: int, fps_est: float | None = None):
        now = time.time()
        if (now - self._last_ui_t) < self._ui_every_sec:
            return
        self._last_ui_t = now

        if self.total_frames > 0:
            self.bar.config(mode="determinate")
            self.bar["maximum"] = self.total_frames
            self.bar["value"] = min(frame_idx, self.total_frames)

            pct = (frame_idx / self.total_frames) * 100.0
            self.label.config(text=f"Processing... {pct:.1f}%")
            if fps_est and fps_est > 0:
                remaining = max(0, self.total_frames - frame_idx)
                eta_sec = remaining / fps_est
                self.detail.config(
                    text=f"{frame_idx}/{self.total_frames} frames  |  {fps_est:.1f} fps  |  ETA ~ {eta_sec:.1f}s"
                )
            else:
                self.detail.config(text=f"{frame_idx}/{self.total_frames} frames")
        else:
            self.label.config(text="Processing... (frame count unknown)")
            if fps_est and fps_est > 0:
                self.detail.config(text=f"{frame_idx} frames  |  {fps_est:.1f} fps")
            else:
                self.detail.config(text=f"{frame_idx} frames")

        self.root.update_idletasks()
        self.root.update()

    def done(self, out_path: str, frame_idx: int):
        try:
            self.bar.stop()
        except Exception:
            pass

        self.label.config(text="Done ✅")
        self.detail.config(text=f"Saved: {out_path}\nFrames written: {frame_idx}")
        self.bar.config(mode="determinate")
        self.bar["value"] = self.bar["maximum"]

        self.root.update_idletasks()
        self.root.update()

        self.root.after(1200, self.root.destroy)
        self.root.mainloop()


def main():
    video_path = pick_video_file()
    if not video_path:
        print("No file selected. Exiting.")
        return

    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    fps = clamp_fps(cap.get(cv2.CAP_PROP_FPS), fallback=30.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if w <= 0 or h <= 0:
        cap.release()
        raise RuntimeError("Could not read video width/height. Is the file valid?")

    out_path = make_output_path(video_path)
    results_log_path = make_results_log_path(video_path)

    # ---- Redirect stdout to results-only logger (still keep console prints) ----
    old_stdout = sys.stdout
    results_logger = ResultsOnlyLogger(results_log_path, keep_console=True)
    sys.stdout = results_logger

    pipeline = ComputerVision()
    toggles = {"drone": True, "angle": True, "distance": True, "led": True}

    writer, actual_out_path = try_create_writer(out_path, fps, w, h)

    # These prints WILL show in console, but will NOT be written to the log
    # (because the logger only accepts lines starting with "Iteration-")
    print(f"Input : {video_path}")
    print(f"Output: {actual_out_path}")
    print(f"FPS   : {fps:.2f}")
    print(f"Frames: {total if total > 0 else '(unknown)'}")
    print(f"Results log: {results_log_path}")

    ui = ProgressUI(total_frames=total)

    frame_idx = 0
    t0 = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # ---- This will print Iteration-... lines inside ComputerVision,
            #      which WILL be captured into results_log_path.
            out_frame = pipeline.process(frame, toggles)

            if out_frame.shape[1] != w or out_frame.shape[0] != h:
                out_frame = cv2.resize(out_frame, (w, h), interpolation=cv2.INTER_LINEAR)

            writer.write(out_frame)
            frame_idx += 1

            dt = time.time() - t0
            fps_est = (frame_idx / dt) if dt > 0 else None
            ui.update(frame_idx, fps_est=fps_est)

    finally:
        cap.release()
        writer.release()

        # Restore stdout and close logger cleanly
        sys.stdout = old_stdout
        results_logger.close()

    print(f"Done. Saved processed video. Frames written: {frame_idx}")
    print(f"Results-only log saved to: {results_log_path}")
    ui.done(actual_out_path, frame_idx)


if __name__ == "__main__":
    main()