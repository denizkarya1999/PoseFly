# ui.py
import ctypes
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2

ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("posefly.app.v1")

class PipelineUI(tk.Tk):
    """
    Fully UI-based controller:
      - checkboxes for inference modules + saving
      - start/stop buttons
      - drives the backend each frame using Tkinter's event loop (after)
      - closing the OpenCV window with X stops processing and DOES NOT relaunch
    """
    WINDOW_NAME = "PoseFly - Drone Tracker"

    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.iconbitmap("assets/posefly.ico")
        self.title("PoseFly - AI-enhanced Drone-to-Drone Communication System")
        self.geometry("680x360")
        self.resizable(False, False)

        # State
        self.running = False

        # UI variables (tickboxes)
        self.var_drone = tk.BooleanVar(value=True)
        self.var_angle = tk.BooleanVar(value=True)
        self.var_distance = tk.BooleanVar(value=True)
        self.var_led = tk.BooleanVar(value=True)
        self.var_save_video = tk.BooleanVar(value=True)

        self.var_camera_index = tk.IntVar(value=0)
        self.var_fps = tk.StringVar(value="10.0")
        self.var_output_path = tk.StringVar(value="results/full_pipeline_results/posefly_results2.mp4")

        # Build layout
        self._build()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build(self):
        pad = 10

        frm = ttk.Frame(self)
        frm.pack(fill="both", expand=True, padx=pad, pady=pad)

        # Inference toggles
        box_inf = ttk.LabelFrame(frm, text="Inference")
        box_inf.pack(fill="x", pady=(0, pad))

        ttk.Checkbutton(box_inf, text="Drone Detection", variable=self.var_drone)\
            .grid(row=0, column=0, sticky="w", padx=pad, pady=5)
        ttk.Checkbutton(box_inf, text="Angle Inference", variable=self.var_angle)\
            .grid(row=1, column=0, sticky="w", padx=pad, pady=5)
        ttk.Checkbutton(box_inf, text="Distance Inference", variable=self.var_distance)\
            .grid(row=2, column=0, sticky="w", padx=pad, pady=5)
        ttk.Checkbutton(box_inf, text="LED ID Inference", variable=self.var_led)\
            .grid(row=3, column=0, sticky="w", padx=pad, pady=5)

        # Saving + settings
        box_io = ttk.LabelFrame(frm, text="Settings")
        box_io.pack(fill="x", pady=(0, pad))

        ttk.Checkbutton(box_io, text="Save Video", variable=self.var_save_video)\
            .grid(row=0, column=0, sticky="w", padx=pad, pady=5)

        ttk.Label(box_io, text="Camera Index:").grid(row=1, column=0, sticky="w", padx=pad, pady=5)
        ttk.Spinbox(box_io, from_=0, to=10, textvariable=self.var_camera_index, width=6)\
            .grid(row=1, column=1, sticky="w", pady=5)

        ttk.Label(box_io, text="FPS:").grid(row=1, column=2, sticky="e", padx=(pad, 5), pady=5)
        ttk.Entry(box_io, textvariable=self.var_fps, width=8)\
            .grid(row=1, column=3, sticky="w", pady=5)

        ttk.Label(box_io, text="Output Path:").grid(row=2, column=0, sticky="w", padx=pad, pady=5)
        ent_path = ttk.Entry(box_io, textvariable=self.var_output_path, width=45)
        ent_path.grid(row=2, column=1, columnspan=3, sticky="w", pady=5)
        ttk.Button(box_io, text="Browse", command=self.browse_output)\
            .grid(row=2, column=4, padx=pad, pady=5)

        # Controls
        box_ctrl = ttk.LabelFrame(frm, text="Controls")
        box_ctrl.pack(fill="x")

        ttk.Button(box_ctrl, text="Start", command=self.start).grid(row=0, column=0, padx=pad, pady=pad)
        ttk.Button(box_ctrl, text="Stop", command=self.stop).grid(row=0, column=1, padx=pad, pady=pad)
        ttk.Button(box_ctrl, text="Quit", command=self.on_close).grid(row=0, column=2, padx=pad, pady=pad)

        self.lbl_status = ttk.Label(box_ctrl, text="Status: Idle")
        self.lbl_status.grid(row=0, column=3, padx=pad, pady=pad, sticky="w")

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Select output video file",
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4"), ("All Files", "*.*")]
        )
        if path:
            self.var_output_path.set(path)

    def _get_fps(self) -> float:
        try:
            fps = float(self.var_fps.get().strip())
            return fps if fps > 0 else 10.0
        except Exception:
            return 10.0

    def _get_toggles(self) -> dict:
        return {
            "drone": self.var_drone.get(),
            "angle": self.var_angle.get(),
            "distance": self.var_distance.get(),
            "led": self.var_led.get(),
        }

    def _window_closed(self) -> bool:
        """
        True if the OpenCV window is closed (or doesn't exist).
        """
        try:
            return cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True

    def start(self):
        if self.running:
            return

        try:
            self.backend.open_camera(camera_index=self.var_camera_index.get(), use_dshow=True)
            self.backend.apply_camera_settings_led_id()
        except Exception as e:
            messagebox.showerror("Start Failed", str(e))
            return

        # Create OpenCV window once (normal GUI => only X)
        try:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        except Exception:
            pass

        self.running = True
        self.lbl_status.config(text="Status: Running")
        self._tick()

    def stop(self):
        if not self.running:
            return

        self.running = False
        self.lbl_status.config(text="Status: Stopped")

        # cleanup backend + windows
        try:
            self.backend.release_writer()
        except Exception:
            pass
        try:
            self.backend.release_camera()
        except Exception:
            pass

        # Close only this window (avoid side effects)
        try:
            cv2.destroyWindow(self.WINDOW_NAME)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _tick(self):
        if not self.running:
            return

        # If user closed the OpenCV window, stop BEFORE imshow recreates it
        if self._window_closed():
            self.stop()
            return

        ret, frame = self.backend.read_frame()
        if not ret:
            self.stop()
            return

        frame = self.backend.process_frame(frame, self._get_toggles())

        # Show frame (safe because we already checked window is open)
        cv2.imshow(self.WINDOW_NAME, frame)

        save_video = self.var_save_video.get()
        output_path = self.var_output_path.get().strip()
        fps = self._get_fps()

        try:
            self.backend.write_frame_if_enabled(frame, save_video, output_path, fps)
        except Exception as e:
            self.var_save_video.set(False)
            messagebox.showwarning("Video Writer Error", f"{e}\n\nVideo saving has been turned OFF.")
            try:
                self.backend.release_writer()
            except Exception:
                pass

        # Keep OpenCV responsive (no ESC handling)
        cv2.waitKey(1)

        self.after(1, self._tick)

    def on_close(self):
        self.stop()
        self.destroy()