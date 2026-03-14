import os
import sys
import time
import threading
import platform
import traceback
import math
import tempfile
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from backend import PipelineBackend

ICON_PATH = "/home/denizkaryaacikbas/StudioProjects/PoseFly/PoseFly Desktop App/PoseFly.WinForms/assets/PoseFly.png"


# ---------------------------
# Results-only logging
# ---------------------------
class ResultsOnlyLogger:
    """
    Redirects sys.stdout to a file but only writes lines that look like:
      Iteration-123: ...
    Everything else is dropped.
    """

    def __init__(self, file_path: str, keep_console: bool = True):
        self.file_path = file_path
        self.file = open(file_path, "w", encoding="utf-8")
        self._stdout = sys.stdout
        self.keep_console = keep_console
        self._buf = ""

    def write(self, s: str):
        if self.keep_console:
            self._stdout.write(s)

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
        if self._buf.strip().startswith("Iteration-"):
            self.file.write(self._buf.strip() + "\n")
        self.file.flush()
        self.file.close()


def make_results_log_path(output_path: str) -> str:
    """
    Build a results-only txt path from the chosen output path.
    Example:
      results/video.mp4 -> results/video_results_only.txt
    """
    if output_path and str(output_path).strip():
        base, _ = os.path.splitext(output_path)
        return base + "_results_only.txt"

    # fallback if no output video path is used
    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join("results", f"posefly_live_{ts}_results_only.txt")


def clamp_int(x, lo, hi, default):
    try:
        v = int(float(x))
        return max(lo, min(hi, v))
    except Exception:
        return default


def clamp_float(x, lo, hi, default):
    try:
        v = float(x)
        return max(lo, min(hi, v))
    except Exception:
        return default


def apply_rolling_shutter(frame_bgr: np.ndarray, iso: int, shutter_hz: float) -> np.ndarray:
    if frame_bgr is None:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    if h <= 1:
        return frame_bgr

    row_time = 1.0 / max(1, int(shutter_hz))
    out = frame_bgr.copy()

    # ISO -> gain (log mapping)
    iso_f = float(max(1, iso))
    t_iso = (math.log(iso_f) - math.log(50.0)) / (math.log(6400.0) - math.log(50.0))
    t_iso = max(0.0, min(1.0, t_iso))
    gain = 2.0 + t_iso * 18.0  # ~2..20

    # shutter_hz -> exposure proxy (higher Hz => darker)
    sh_f = float(max(1, shutter_hz))
    t_sh = (math.log(sh_f) - math.log(5.0)) / (math.log(6000.0) - math.log(5.0))
    t_sh = max(0.0, min(1.0, t_sh))

    exposure = -10.0 + (1.0 - t_sh) * 6.0
    exposure_scale = 2.0 ** (exposure / 2.0)

    brightness = 95.0 + (1.0 - t_sh) * 15.0
    brightness_scale = brightness / 100.0

    # OOK stripe model (row-time based)
    led_freq_hz = 2000.0
    duty = 0.5
    contrast = 0.90

    y = np.arange(h, dtype=np.float32)
    t = y * row_time
    phase = (t * led_freq_hz) % 1.0
    on = (phase < duty).astype(np.float32)

    row_gain = (1.0 - contrast) + contrast * on
    row_gain = row_gain[:, None]  # (h,1)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]

    v *= gain
    v *= exposure_scale
    v *= brightness_scale
    v *= row_gain

    hsv[:, :, 2] = np.clip(v, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def _probe_camera(index: int, backend: int) -> tuple[bool, tuple[int, int] | None]:
    cap = None
    try:
        cap = cv2.VideoCapture(int(index), int(backend))
        if not cap.isOpened():
            return False, None
        ok, frame = cap.read()
        if not ok or frame is None:
            return False, None
        h, w = frame.shape[:2]
        return True, (w, h)
    except Exception:
        return False, None
    finally:
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


def list_available_cameras(max_index: int = 20) -> list[dict]:
    sysname = platform.system()
    if sysname == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        backend_names = {cv2.CAP_DSHOW: "DShow", cv2.CAP_MSMF: "MSMF", cv2.CAP_ANY: "Any"}
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY, cv2.CAP_GSTREAMER]
        backend_names = {cv2.CAP_V4L2: "V4L2", cv2.CAP_ANY: "Any", cv2.CAP_GSTREAMER: "GStreamer"}

    found = []
    for i in range(0, max_index + 1):
        for b in backends:
            ok, wh = _probe_camera(i, b)
            if ok:
                wh_txt = f"{wh[0]}x{wh[1]}" if wh else "?"
                lbl = f"Camera {i}  ({backend_names.get(b, str(b))}, {wh_txt})"
                found.append({"index": i, "label": lbl})
                break
    return found


class PoseFlyGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PoseFly Pipeline")

        self._tmp_icon_png = None
        self._set_window_icon()

        self.backend = PipelineBackend()

        self._running = False
        self._thread = None
        self._last_frame_ts = 0.0
        self._tk_img = None

        # Rolling-shutter auto-apply bookkeeping
        self._rs_after_id = None
        self._last_rs_applied = (None, None)

        # Cached RS values used for GUI-side effect
        self._rs_iso_cached = 250
        self._rs_shz_cached = 1000.0

        # Frame counter for periodic re-apply
        self._frame_count = 0

        # Camera list state
        self._cams = []
        self._cam_label_to_index = {}

        # Results logging / trajectory
        self._results_log_path = None
        self._trajectory_launched = False

        # UI state
        self.cam_choice_var = tk.StringVar(value="(scan to list cameras)")
        self.cam_index_var = tk.IntVar(value=0)
        self.use_dshow_var = tk.BooleanVar(value=(platform.system() == "Windows"))

        self.fps_var = tk.StringVar(value="60")
        self.save_var = tk.BooleanVar(value=True)
        self.out_path_var = tk.StringVar(value=os.path.join("results", "posefly_results.mp4"))

        self.iso_var = tk.StringVar(value="250")
        self.shutter_var = tk.StringVar(value="1000")
        self.use_rs_var = tk.BooleanVar(value=True)

        self.tog_drone = tk.BooleanVar(value=True)
        self.tog_angle = tk.BooleanVar(value=True)
        self.tog_distance = tk.BooleanVar(value=True)
        self.tog_led = tk.BooleanVar(value=True)
        self.tog_speed = tk.BooleanVar(value=True)

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.refresh_cameras()
        self._ui_tick()

    # --------------------------------------------------------
    # WINDOW ICON (ICO on Linux: convert to temp PNG and use iconphoto)
    # --------------------------------------------------------
    def _set_window_icon(self):
        if not os.path.exists(ICON_PATH):
            return
        try:
            sysname = platform.system()
            if sysname == "Windows":
                self.root.iconbitmap(ICON_PATH)
                return

            img = Image.open(ICON_PATH)

            fd, tmp_png = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            self._tmp_icon_png = tmp_png
            img.save(tmp_png, format="PNG")

            icon = tk.PhotoImage(file=tmp_png)
            self.root.iconphoto(True, icon)
            self._icon_ref = icon
        except Exception:
            pass

    # --------------------------------------------------------
    # Camera selection helpers
    # --------------------------------------------------------
    def refresh_cameras(self):
        try:
            self.status.config(text="Status: Scanning cameras...")
            self.root.update_idletasks()

            cams = list_available_cameras(max_index=20)
            self._cams = cams
            self._cam_label_to_index = {c["label"]: c["index"] for c in cams}

            menu = self.cam_dropdown["menu"]
            menu.delete(0, "end")

            if not cams:
                self.cam_choice_var.set("(no cameras found)")
                menu.add_command(label="(no cameras found)",
                                 command=lambda: self.cam_choice_var.set("(no cameras found)"))
                self.status.config(text="Status: No cameras found (try permissions / close other apps).")
                return

            for c in cams:
                lbl = c["label"]
                menu.add_command(label=lbl, command=lambda v=lbl: self.cam_choice_var.set(v))

            self.cam_choice_var.set(cams[0]["label"])
            self.cam_index_var.set(int(cams[0]["index"]))
            self.status.config(text=f"Status: Found {len(cams)} camera(s).")
        except Exception as e:
            self.status.config(text=f"Status: Camera scan error: {e}")

    def _selected_camera_index(self) -> int:
        choice = self.cam_choice_var.get()
        if choice in self._cam_label_to_index:
            idx = int(self._cam_label_to_index[choice])
            self.cam_index_var.set(idx)
            return idx
        return int(self.cam_index_var.get())

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=10, pady=10)

        # Camera row
        row0 = tk.Frame(top)
        row0.pack(fill="x")

        tk.Label(row0, text="Camera:").pack(side="left")
        self.cam_dropdown = tk.OptionMenu(row0, self.cam_choice_var, "(scan to list cameras)")
        self.cam_dropdown.config(width=30)
        self.cam_dropdown.pack(side="left", padx=6)

        self.btn_refresh = tk.Button(row0, text="Refresh", command=self.refresh_cameras)
        self.btn_refresh.pack(side="left", padx=6)

        tk.Label(row0, text="Index:").pack(side="left", padx=(12, 0))
        tk.Spinbox(row0, from_=0, to=20, width=5, textvariable=self.cam_index_var).pack(side="left", padx=6)

        self.chk_dshow = tk.Checkbutton(row0, text="Use DirectShow (Windows)", variable=self.use_dshow_var)
        self.chk_dshow.pack(side="left", padx=10)

        if platform.system() != "Windows":
            self.use_dshow_var.set(False)
            self.chk_dshow.configure(state="disabled")

        # Rolling shutter row
        row1 = tk.Frame(top)
        row1.pack(fill="x", pady=(8, 0))

        tk.Label(row1, text="ISO:").pack(side="left")
        self.iso_entry = tk.Entry(row1, textvariable=self.iso_var, width=10)
        self.iso_entry.pack(side="left", padx=6)

        tk.Label(row1, text="Shutter (Hz):").pack(side="left")
        self.shutter_entry = tk.Entry(row1, textvariable=self.shutter_var, width=10)
        self.shutter_entry.pack(side="left", padx=6)

        tk.Checkbutton(row1, text="Rolling Shutter Effect (GUI)", variable=self.use_rs_var).pack(side="left", padx=10)

        self.iso_entry.bind("<KeyRelease>", self._schedule_apply_rs)
        self.shutter_entry.bind("<KeyRelease>", self._schedule_apply_rs)
        self.iso_entry.bind("<FocusOut>", self._schedule_apply_rs)
        self.shutter_entry.bind("<FocusOut>", self._schedule_apply_rs)

        self.btn_apply_rs = tk.Button(row1, text="Apply ISO/Hz", command=self.apply_rs, state="disabled")
        self.btn_apply_rs.pack(side="left", padx=10)

        # Output row
        row2 = tk.Frame(top)
        row2.pack(fill="x", pady=(8, 0))

        tk.Label(row2, text="FPS:").pack(side="left")
        tk.Entry(row2, textvariable=self.fps_var, width=8).pack(side="left", padx=6)

        tk.Checkbutton(row2, text="Save video", variable=self.save_var).pack(side="left", padx=10)

        tk.Label(row2, text="Output:").pack(side="left")
        tk.Entry(row2, textvariable=self.out_path_var).pack(side="left", padx=6, fill="x", expand=True)

        tk.Button(row2, text="Browse…", command=self.choose_output).pack(side="left", padx=6)

        # Vision toggles
        row3 = tk.LabelFrame(top, text="Vision toggles")
        row3.pack(fill="x", pady=(10, 0))

        tk.Checkbutton(row3, text="Drone", variable=self.tog_drone).pack(side="left", padx=6)
        tk.Checkbutton(row3, text="Angle", variable=self.tog_angle).pack(side="left", padx=6)
        tk.Checkbutton(row3, text="Distance", variable=self.tog_distance).pack(side="left", padx=6)
        tk.Checkbutton(row3, text="LED", variable=self.tog_led).pack(side="left", padx=6)
        tk.Checkbutton(row3, text="Speed", variable=self.tog_speed).pack(side="left", padx=6)

        # Start/Stop
        row4 = tk.Frame(top)
        row4.pack(fill="x", pady=(10, 0))

        self.btn_start = tk.Button(row4, text="Start", command=self.start)
        self.btn_start.pack(side="left")

        self.btn_stop = tk.Button(row4, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.pack(side="left", padx=8)

        # Status + preview
        self.status = tk.Label(self.root, text="Status: Idle", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0, 6))

        self.preview = tk.Label(self.root)
        self.preview.pack(padx=10, pady=10)

    # --------------------------------------------------------
    # Output chooser
    # --------------------------------------------------------
    def choose_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi"), ("All files", "*.*")],
        )
        if path:
            self.out_path_var.set(path)

    # --------------------------------------------------------
    # Rolling shutter handling
    # --------------------------------------------------------
    def _read_rs_inputs(self) -> tuple[int, float]:
        iso = clamp_int(self.iso_var.get(), 50, 6400, 250)
        shz = clamp_float(self.shutter_var.get(), 5.0, 6000.0, 1000.0)
        return iso, shz

    def _schedule_apply_rs(self, _event=None):
        iso, shz = self._read_rs_inputs()
        self._rs_iso_cached, self._rs_shz_cached = iso, shz

        if not self._running:
            return

        if self._rs_after_id:
            try:
                self.root.after_cancel(self._rs_after_id)
            except Exception:
                pass
        self._rs_after_id = self.root.after(200, self._apply_rs_if_changed)

    def _apply_rs_if_changed(self, force=False):
        iso, shz = self._read_rs_inputs()
        self._rs_iso_cached, self._rs_shz_cached = iso, shz

        if not self._running:
            return

        if (not force) and self._last_rs_applied == (iso, shz):
            return

        self._last_rs_applied = (iso, shz)
        try:
            self.backend.set_rollingshutter(iso, shz)
            self.status.config(text=f"Status: Applied ISO={iso}, ShutterHz={shz}")
        except Exception as e:
            self.status.config(text=f"Status: WARN RS backend apply failed: {e}")

    def apply_rs(self):
        self._apply_rs_if_changed(force=True)

    # --------------------------------------------------------
    # Trajectory launcher
    # --------------------------------------------------------
    def _launch_trajectory_algorithm(self, log_file_path: str):
        if self._trajectory_launched:
            return

        if not log_file_path or not os.path.exists(log_file_path):
            self._set_status_threadsafe("Status: Trajectory skipped (log file not found).")
            return

        if os.path.getsize(log_file_path) <= 0:
            self._set_status_threadsafe("Status: Trajectory skipped (log file is empty).")
            return

        here = os.path.dirname(os.path.abspath(__file__))
        trajectory_script = os.path.join(here, "videoprocessing", "tools", "trajectory_algorithm.py")

        if not os.path.isfile(trajectory_script):
            self._set_status_threadsafe(f"Status: Trajectory script not found: {trajectory_script}")
            return

        cmd = [sys.executable, trajectory_script, log_file_path]

        try:
            subprocess.Popen(cmd)
            self._trajectory_launched = True
            self._set_status_threadsafe("Status: Stopped | Trajectory launched")
        except Exception as e:
            self._set_status_threadsafe(f"Status: Failed to launch trajectory: {e}")

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------
    def start(self):
        if self._running:
            return

        cam_idx = self._selected_camera_index()
        use_dshow = bool(self.use_dshow_var.get()) if platform.system() == "Windows" else False
        fps = clamp_float(self.fps_var.get(), 1.0, 240.0, 60.0)
        out_path = str(self.out_path_var.get()).strip()
        save_video = bool(self.save_var.get())

        if save_video and not out_path:
            messagebox.showwarning("Missing output path", "Choose an output path or disable Save video.")
            return

        iso, shz = self._read_rs_inputs()
        self._rs_iso_cached, self._rs_shz_cached = iso, shz

        # Prepare results-only log path
        self._results_log_path = make_results_log_path(out_path)
        self._trajectory_launched = False

        os.makedirs(os.path.dirname(self._results_log_path) or ".", exist_ok=True)

        self._running = True
        self._frame_count = 0
        self._last_rs_applied = (None, None)

        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_apply_rs.config(state="normal")
        self.btn_refresh.config(state="disabled")
        self.cam_dropdown.config(state="disabled")
        self.status.config(text=f"Status: Starting... | Log: {self._results_log_path}")

        self._thread = threading.Thread(
            target=self._pipeline_loop,
            args=(cam_idx, use_dshow, fps, out_path, save_video, self._results_log_path),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._running = False
        self.status.config(text="Status: Stopping...")

    def _pipeline_loop(self, cam_idx, use_dshow, fps, out_path, save_video, results_log_path):
        old_stdout = sys.stdout
        results_logger = None

        try:
            results_logger = ResultsOnlyLogger(results_log_path, keep_console=True)
            sys.stdout = results_logger
        except Exception as e:
            sys.stdout = old_stdout
            self._set_status_threadsafe(f"Status: WARN could not start results logger: {e}")
            results_logger = None

        try:
            self.backend.open_camera(camera_index=cam_idx, use_dshow=use_dshow)
            self.backend.apply_camera_settings_led_id()
        except Exception as e:
            print(traceback.format_exc())
            self._set_status_threadsafe(f"Status: Camera ERROR: {e}")
            self._running = False
            self._set_buttons_threadsafe(False)
            if results_logger is not None:
                try:
                    sys.stdout = old_stdout
                    results_logger.close()
                except Exception:
                    pass
            return

        self.root.after(0, lambda: self._apply_rs_if_changed(force=True))
        target_dt = 1.0 / max(fps, 1.0)

        try:
            while self._running:
                t0 = time.time()
                try:
                    ok, frame = self.backend.read_frame()
                    if not ok or frame is None:
                        self._set_status_threadsafe("Status: ERROR failed to read frame")
                        break

                    self._frame_count += 1

                    # periodic backend apply (in case camera resets)
                    if (self._frame_count % 30) == 0:
                        self.root.after(0, lambda: self._apply_rs_if_changed(force=True))

                    toggles = {
                        "drone": bool(self.tog_drone.get()),
                        "angle": bool(self.tog_angle.get()),
                        "distance": bool(self.tog_distance.get()),
                        "led": bool(self.tog_led.get()),
                        "speed": bool(self.tog_speed.get()),
                    }
                    out = self.backend.process_frame(frame, toggles)
                    if isinstance(out, np.ndarray):
                        frame = out
                    elif isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], np.ndarray):
                        frame = out[1]

                    # GUI rolling shutter effect
                    if self.use_rs_var.get():
                        frame = apply_rolling_shutter(frame, self._rs_iso_cached, self._rs_shz_cached)

                    if save_video:
                        try:
                            self.backend.write_frame_if_enabled(frame, True, out_path, fps)
                        except Exception as e:
                            save_video = False
                            self._set_status_threadsafe(f"Status: WARN writer disabled: {e}")
                            try:
                                self.backend.release_writer()
                            except Exception:
                                pass

                    self._show_frame_threadsafe(frame)

                except Exception as e:
                    print(traceback.format_exc())
                    self._set_status_threadsafe(f"Status: ERROR {e}")
                    break

                dt = time.time() - t0
                if dt < target_dt:
                    time.sleep(target_dt - dt)

        finally:
            try:
                self.backend.release_camera()
            except Exception:
                pass
            try:
                self.backend.release_writer()
            except Exception:
                pass

            try:
                sys.stdout = old_stdout
            except Exception:
                pass

            try:
                if results_logger is not None:
                    results_logger.close()
            except Exception:
                pass

        self._running = False
        self._set_buttons_threadsafe(False)

        # Launch trajectory once after inference fully stops
        try:
            self._launch_trajectory_algorithm(results_log_path)
        except Exception as e:
            self._set_status_threadsafe(f"Status: Stopped | Trajectory launch error: {e}")
            return

        if not self._trajectory_launched:
            self._set_status_threadsafe("Status: Stopped")

    # --------------------------------------------------------
    # Thread-safe UI helpers
    # --------------------------------------------------------
    def _set_status_threadsafe(self, text: str):
        self.root.after(0, lambda: self.status.config(text=text))

    def _set_buttons_threadsafe(self, running: bool):
        def _do():
            self.btn_start.config(state="disabled" if running else "normal")
            self.btn_stop.config(state="normal" if running else "disabled")
            self.btn_apply_rs.config(state="normal" if running else "disabled")
            self.btn_refresh.config(state="disabled" if running else "normal")
            self.cam_dropdown.config(state="disabled" if running else "normal")
        self.root.after(0, _do)

    def _show_frame_threadsafe(self, frame_bgr):
        now = time.time()
        if now - self._last_frame_ts < 1.0 / 60.0:
            return
        self._last_frame_ts = now

        def _do():
            try:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)

                max_w, max_h = 1100, 650
                iw, ih = img.size
                scale = min(max_w / iw, max_h / ih, 1.0)
                if scale < 1.0:
                    img = img.resize((int(iw * scale), int(ih * scale)))

                self._tk_img = ImageTk.PhotoImage(img)
                self.preview.configure(image=self._tk_img)
            except Exception:
                pass

        self.root.after(0, _do)

    def _ui_tick(self):
        if platform.system() != "Windows" and self.use_dshow_var.get():
            self.use_dshow_var.set(False)
        self.root.after(250, self._ui_tick)

    def on_close(self):
        self.stop()
        try:
            time.sleep(0.05)
        except Exception:
            pass
        try:
            self.backend.release_writer()
        except Exception:
            pass
        try:
            self.backend.release_camera()
        except Exception:
            pass

        # cleanup temp icon file if created
        try:
            if self._tmp_icon_png and os.path.exists(self._tmp_icon_png):
                os.remove(self._tmp_icon_png)
        except Exception:
            pass

        self.root.destroy()


def main():
    root = tk.Tk()
    app = PoseFlyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()