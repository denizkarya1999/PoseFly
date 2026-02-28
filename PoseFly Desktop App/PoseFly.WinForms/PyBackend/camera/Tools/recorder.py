# ui_recorder.py
# Changes added:
# - Camera selection UI (dropdown) + "Refresh" button
# - Cross-platform camera enumeration (Windows: DShow/MSMF, Linux: V4L2/Any/GStreamer)
# - Keeps your existing “Camera Index” spinbox too (advanced/manual override)
# - Compatibility behavior remains: only use DShow on Windows

import time
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import math
import numpy as np

import os, sys
import platform
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from camera import Camera

DEFAULT_FPS = 60.0


def apply_rolling_shutter(frame, iso, shutter_hz):
    h, w, _ = frame.shape
    row_time = 1.0 / max(1, int(shutter_hz))
    out = frame.copy()

    # --- Camera-like mapping (matches camera.py) ---

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

    # --- OOK stripe model (row-time based) ---
    led_freq_hz = 2000
    duty = 0.5
    contrast = 0.90

    y = np.arange(h, dtype=np.float32)
    t = y * row_time
    phase = (t * led_freq_hz) % 1.0
    on = (phase < duty).astype(np.float32)

    row_gain = (1.0 - contrast) + contrast * on
    row_gain = row_gain[:, None]

    # Apply everything in HSV value channel
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
    """
    Try to open/read one frame from (index, backend).
    Returns: (ok, (w,h) or None)
    """
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


def list_available_cameras(max_index: int = 10) -> list[dict]:
    """
    Cross-platform camera enumeration using OpenCV probing.
    Returns list of dicts: {"index": i, "label": "...", "backend": backend_int}
    """
    sysname = platform.system()

    # Try multiple backends for discovery (ordered by what usually works best)
    if sysname == "Windows":
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        backend_names = {
            cv2.CAP_DSHOW: "DShow",
            cv2.CAP_MSMF: "MSMF",
            cv2.CAP_ANY: "Any",
        }
    else:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY, cv2.CAP_GSTREAMER]
        backend_names = {
            cv2.CAP_V4L2: "V4L2",
            cv2.CAP_ANY: "Any",
            cv2.CAP_GSTREAMER: "GStreamer",
        }

    found = []
    seen_indices = set()

    # Probe indices; first successful backend "claims" that index for the dropdown
    for i in range(0, max_index + 1):
        if i in seen_indices:
            continue
        for b in backends:
            ok, wh = _probe_camera(i, b)
            if ok:
                seen_indices.add(i)
                w_h_str = f"{wh[0]}x{wh[1]}" if wh else "?"
                bname = backend_names.get(b, str(b))
                label = f"Camera {i}  ({bname}, {w_h_str})"
                found.append({"index": i, "label": label, "backend": b})
                break

    return found


class RecorderUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PoseFly Webcam Recorder")

        self.cam = Camera()
        self.running = True

        # recording state
        self.record_enabled = False
        self.output_path = ""
        self.fps = float(DEFAULT_FPS)

        # preview bookkeeping
        self._tk_img = None

        # camera list state
        self._cams = []
        self._cam_label_to_index = {}

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Initial camera scan
        self.refresh_cameras()

    def _build_ui(self):
        panel = tk.Frame(self.root)
        panel.pack(fill="x", padx=10, pady=10)

        # --- Camera selection dropdown (NEW) ---
        tk.Label(panel, text="Camera:").grid(row=0, column=0, sticky="w")
        self.cam_choice_var = tk.StringVar(value="(scan to list cameras)")
        self.cam_dropdown = tk.OptionMenu(panel, self.cam_choice_var, "(scan to list cameras)")
        self.cam_dropdown.config(width=28)
        self.cam_dropdown.grid(row=0, column=1, sticky="w", padx=6)

        self.btn_refresh = tk.Button(panel, text="Refresh", command=self.refresh_cameras)
        self.btn_refresh.grid(row=0, column=2, padx=8)

        # Manual camera index (kept; useful when dropdown misses a device)
        tk.Label(panel, text="Camera Index:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.cam_index_var = tk.IntVar(value=0)
        tk.Spinbox(panel, from_=0, to=20, width=5, textvariable=self.cam_index_var)\
            .grid(row=1, column=1, sticky="w", padx=6, pady=(6, 0))

        self.btn_open = tk.Button(panel, text="Open Camera", command=self.open_camera)
        self.btn_open.grid(row=1, column=2, padx=8, pady=(6, 0))

        self.btn_close = tk.Button(panel, text="Close Camera", command=self.close_camera, state="disabled")
        self.btn_close.grid(row=1, column=3, pady=(6, 0))

        # ISO (typed)
        tk.Label(panel, text="ISO:").grid(row=2, column=0, sticky="w", pady=(10, 0))
        self.iso_var = tk.StringVar(value=str(self.cam.iso))
        self.iso_entry = tk.Entry(panel, textvariable=self.iso_var, width=12)
        self.iso_entry.grid(row=2, column=1, sticky="w", pady=(10, 0))

        # Shutter Hz (typed)
        tk.Label(panel, text="Shutter (Hz):").grid(row=3, column=0, sticky="w")
        self.shutter_var = tk.StringVar(value=str(self.cam.shutter_hz))
        self.shutter_entry = tk.Entry(panel, textvariable=self.shutter_var, width=12)
        self.shutter_entry.grid(row=3, column=1, sticky="w")

        # Apply button
        self.btn_apply_rs = tk.Button(panel, text="Apply ISO/Hz", command=self.apply_rs, state="disabled")
        self.btn_apply_rs.grid(row=2, column=2, rowspan=2, padx=8, sticky="ns")

        # FPS (typed)
        tk.Label(panel, text="Record FPS:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.fps_var = tk.StringVar(value=str(self.fps))
        tk.Entry(panel, textvariable=self.fps_var, width=12).grid(row=4, column=1, sticky="w", pady=(8, 0))

        # Output path
        out_row = tk.Frame(self.root)
        out_row.pack(fill="x", padx=10, pady=(0, 8))

        self.btn_choose = tk.Button(out_row, text="Choose Output File", command=self.choose_output)
        self.btn_choose.pack(side="left")

        self.lbl_out = tk.Label(out_row, text="(no output selected)", anchor="w")
        self.lbl_out.pack(side="left", padx=8, fill="x", expand=True)

        # Start/Stop
        ctrl = tk.Frame(self.root)
        ctrl.pack(fill="x", padx=10, pady=(0, 8))

        self.btn_start = tk.Button(ctrl, text="Start Recording", command=self.start_recording, state="disabled")
        self.btn_start.pack(side="left")

        self.btn_stop = tk.Button(ctrl, text="Stop Recording", command=self.stop_recording, state="disabled")
        self.btn_stop.pack(side="left", padx=8)

        # Rolling shutter on/off
        self.use_rs_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ctrl, text="Rolling Shutter Effect", variable=self.use_rs_var)\
            .pack(side="left", padx=8)

        # Status
        self.status = tk.Label(self.root, text="Status: Idle", anchor="w")
        self.status.pack(fill="x", padx=10, pady=(0, 8))

        # Preview
        self.preview = tk.Label(self.root)
        self.preview.pack(padx=10, pady=10)

    # ---------------- Camera list UI (NEW) ----------------
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
                menu.add_command(
                    label="(no cameras found)",
                    command=lambda: self.cam_choice_var.set("(no cameras found)")
                )
                self.status.config(text="Status: No cameras found (try manual index / permissions).")
                return

            # populate menu
            for c in cams:
                lbl = c["label"]
                menu.add_command(label=lbl, command=lambda v=lbl: self.cam_choice_var.set(v))

            # select first by default
            self.cam_choice_var.set(cams[0]["label"])
            self.cam_index_var.set(int(cams[0]["index"]))
            self.status.config(text=f"Status: Found {len(cams)} camera(s).")
        except Exception as e:
            self.status.config(text=f"Status: Camera scan error: {e}")

    # ---------------- Helpers ----------------
    def _get_iso_shz(self):
        iso = int(float(self.iso_var.get()))
        shz = float(self.shutter_var.get())
        iso = max(50, min(6400, iso))
        shz = max(5.0, min(6000.0, shz))
        return iso, shz

    def _selected_camera_index(self) -> int:
        """
        If dropdown matches, use that index; otherwise fall back to Spinbox.
        Also sync spinbox when a dropdown choice exists.
        """
        choice = self.cam_choice_var.get()
        if choice in self._cam_label_to_index:
            idx = int(self._cam_label_to_index[choice])
            self.cam_index_var.set(idx)
            return idx
        return int(self.cam_index_var.get())

    # ---------------- Camera controls ----------------
    def open_camera(self):
        try:
            idx = self._selected_camera_index()

            # ---- COMPATIBILITY: only use DShow on Windows ----
            use_dshow = (platform.system() == "Windows")
            self.cam.open(camera_index=idx, use_dshow=use_dshow)

            self.cam.apply_led_settings()
            self.apply_rs()
        except Exception as e:
            messagebox.showerror("Camera Error", str(e))
            return

        self.btn_open.config(state="disabled")
        self.btn_close.config(state="normal")
        self.btn_apply_rs.config(state="normal")
        self.btn_start.config(state="normal" if self.output_path else "disabled")
        self.btn_refresh.config(state="disabled")
        self.cam_dropdown.config(state="disabled")
        self.status.config(text=f"Status: Camera opened (index={idx})")
        self._update_loop()

    def close_camera(self):
        self.stop_recording()
        self.cam.release()
        self.btn_open.config(state="normal")
        self.btn_close.config(state="disabled")
        self.btn_apply_rs.config(state="disabled")
        self.btn_start.config(state="disabled")
        self.btn_refresh.config(state="normal")
        self.cam_dropdown.config(state="normal")
        self.status.config(text="Status: Camera closed")

    def apply_rs(self):
        if self.cam.cap is None:
            return
        try:
            iso, shz = self._get_iso_shz()
            self.cam.rollingshutter(iso, shz)
            self.status.config(text=f"Status: Applied ISO={iso}, ShutterHz={shz}")
        except Exception:
            messagebox.showerror("Input Error", "Enter valid ISO (50..6400) and Hertz (5..6000).")

    # ---------------- Recording controls ----------------
    def choose_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi"), ("All files", "*.*")]
        )
        if path:
            self.output_path = path
            self.lbl_out.config(text=path)
            if self.cam.cap is not None:
                self.btn_start.config(state="normal")

    def start_recording(self):
        if self.cam.cap is None:
            messagebox.showwarning("Missing camera", "Open the camera first.")
            return
        if not self.output_path:
            self.choose_output()
            if not self.output_path:
                return

        try:
            fps = float(self.fps_var.get())
            if fps <= 0:
                raise ValueError
            self.fps = fps
        except Exception:
            messagebox.showerror("FPS Error", "Please enter a valid FPS number (e.g., 30).")
            return

        self.record_enabled = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.btn_choose.config(state="disabled")
        self.status.config(text="Status: Recording...")

    def stop_recording(self):
        self.record_enabled = False
        try:
            self.cam.write_if_enabled(None, False, self.output_path, self.fps)
        except Exception:
            pass

        self.btn_stop.config(state="disabled")
        self.btn_choose.config(state="normal")
        if self.cam.cap is not None:
            self.btn_start.config(state="normal" if self.output_path else "disabled")
        self.status.config(text="Status: Idle")

    # ---------------- Frame loop ----------------
    def _update_loop(self):
        if not self.running or self.cam.cap is None:
            return

        ok, frame = self.cam.read()
        if not ok or frame is None:
            self.status.config(text="Status: Failed to read frame")
            self.root.after(30, self._update_loop)
            return

        if self.use_rs_var.get():
            try:
                iso, shz = self._get_iso_shz()
                frame = apply_rolling_shutter(frame, iso=iso, shutter_hz=shz)
            except Exception:
                pass

        try:
            self.cam.write_if_enabled(frame, self.record_enabled, self.output_path, self.fps)
        except Exception as e:
            self.status.config(text=f"Status: Writer error: {e}")
            self.record_enabled = False
            self.btn_stop.config(state="disabled")
            self.btn_choose.config(state="normal")
            self.btn_start.config(state="normal")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)

        max_w, max_h = 960, 540
        iw, ih = img.size
        scale = min(max_w / iw, max_h / ih, 1.0)
        if scale < 1.0:
            img = img.resize((int(iw * scale), int(ih * scale)))

        self._tk_img = ImageTk.PhotoImage(img)
        self.preview.configure(image=self._tk_img)

        self.root.after(15, self._update_loop)

    def on_close(self):
        self.running = False
        self.stop_recording()
        self.cam.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = RecorderUI(root)
    root.mainloop()
