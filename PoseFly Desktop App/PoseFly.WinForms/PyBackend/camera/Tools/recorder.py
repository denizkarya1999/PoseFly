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

import os, sys
import platform
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from camera import Camera
from discovery import list_available_cameras

DEFAULT_FPS = 60.0


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
        tk.Label(panel, text="Shutter (1/s):").grid(row=3, column=0, sticky="w")
        self.shutter_var = tk.StringVar(value=str(self.cam.shutter_hz))
        self.shutter_entry = tk.Entry(panel, textvariable=self.shutter_var, width=12)
        self.shutter_entry.grid(row=3, column=1, sticky="w")

        # Apply button
        self.btn_apply_rs = tk.Button(panel, text="Apply camera settings", command=self.apply_rs, state="disabled")
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
        tk.Checkbutton(ctrl, text="Manual camera exposure", variable=self.use_rs_var, command=self.apply_rs)\
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

            cams = list_available_cameras()
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
                menu.add_command(label=lbl, command=lambda v=lbl: self._select_camera(v))

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

    def _select_camera(self, label):
        self.cam_choice_var.set(label)
        self.cam_index_var.set(self._cam_label_to_index[label])

    def _selected_camera_index(self) -> int:
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
            report = self.cam.rollingshutter(iso, shz, enabled=self.use_rs_var.get())
            self.status.config(text=f"Status: {report}")
        except Exception:
            messagebox.showerror("Input Error", "Enter valid ISO (50..6400) and shutter reciprocal (5..6000).")

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
