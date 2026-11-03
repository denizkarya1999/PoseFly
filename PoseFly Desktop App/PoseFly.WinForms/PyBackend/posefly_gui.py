import os
import time
import threading
import platform
import traceback
import math
import queue
import webbrowser
from pathlib import Path
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from backend import PipelineBackend
from videoprocessing.computer_vision import DRONE_MODELS
from latest_frame import LatestFrame
from result_logging import ResultLogWriter
from videoprocessing.drone_overlay import draw_recent_drones, feature_values

from camera.discovery import list_available_cameras

ICON_PATH = Path(__file__).resolve().parent.parent / "assets" / "PoseFly.png"
PROJECT_URL = "https://github.com/denizkarya1999/PoseFly"
BG = "#0b1220"
PANEL = "#111c2e"
FIELD = "#18263a"
TEXT = "#e5edf8"
MUTED = "#a3b3c8"
BORDER = "#31435b"


def default_output_path():
    if platform.system() == "Linux":
        data_home = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share"))
        return str(data_home / "posefly" / "results" / "posefly_results.mp4")
    return os.path.join("results", "posefly_results.mp4")


def default_log_path():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(default_output_path()).parent / f"posefly_detections_{stamp}.csv")


def clamp_int(x, lo, hi, default):
    try:
        v = int(float(x))
        return max(lo, min(hi, v))
    except Exception:
        return default


def clamp_float(x, lo, hi, default):
    try:
        v = float(x)
        if not math.isfinite(v):
            return default
        return max(lo, min(hi, v))
    except Exception:
        return default


class PoseFlyGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("PoseFly Pipeline")

        self._set_window_icon()

        self.backend = PipelineBackend()
        self.backend.cv.log_enabled = False  # Desktop result logging requires opt-in.

        self._closing = False
        self._events = queue.SimpleQueue()
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._display_frame = None
        self._overlay_revision = 0
        self._displayed_overlay = None
        self._detections_model = None
        self._detections = []
        self._features = []
        self._features_pending = False
        self._detections_captured_at = None
        self._detections_shape = None
        self._camera_fps = 0.0
        self._analysis_seconds = 0.0
        self._analysis_finished_at = None
        self._stop_event = threading.Event()
        self._capture_wakeup = threading.Event()
        self._live_fps = 60.0
        self._fps_after_id = None
        self._scan_thread = None
        self._settings = None
        self._running = False
        self._thread = None
        self._ui_after_id = None
        self._tk_img = None

        # Rolling-shutter auto-apply bookkeeping
        self._rs_after_id = None
        self._last_rs_applied = (None, None)
        self._rs_request = 0

        # Cached RS values used for GUI-side effect
        self._rs_iso_cached = 250
        self._rs_shz_cached = 1000.0

        # Frame counter for periodic re-apply
        self._frame_count = 0

        # Camera list state
        self._cams = []
        self._cam_label_to_index = {}

        self._log_writer = None
        self._log_generation = 0
        self._log_state = "off"
        self._log_rows = 0

        # UI state
        self.cam_choice_var = tk.StringVar(value="(scan to list cameras)")
        self.cam_index_var = tk.IntVar(value=0)
        self.use_dshow_var = tk.BooleanVar(value=(platform.system() == "Windows"))

        self.fps_var = tk.StringVar(value="60")
        self.live_fps_var = tk.StringVar(value="60")
        self.save_var = tk.BooleanVar(value=True)
        self.out_path_var = tk.StringVar(value=default_output_path())
        self.log_path_var = tk.StringVar(value=default_log_path())

        self.iso_var = tk.StringVar(value="250")
        self.shutter_var = tk.StringVar(value="1000")
        self.use_rs_var = tk.BooleanVar(value=False)

        self.drone_model_var = tk.StringVar(value="Indoor")
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
    # Portable bundled PNG icon
    # --------------------------------------------------------
    def _set_window_icon(self):
        try:
            self._icon_ref = tk.PhotoImage(file=str(ICON_PATH))
            self.root.iconphoto(True, self._icon_ref)
            self.root.iconname("PoseFly")
        except (OSError, tk.TclError):
            pass

    def refresh_cameras(self):
        if self._running or (self._scan_thread and self._scan_thread.is_alive()):
            return
        self.status.config(text="Status: Scanning cameras...")
        self.btn_refresh.config(state="disabled", text="Scanning…")
        self.btn_start.config(state="disabled")
        self._scan_thread = threading.Thread(target=self._scan_cameras, daemon=True)
        self._scan_thread.start()

    def _scan_cameras(self):
        try:
            self._events.put(("cameras", list_available_cameras()))
        except Exception as exc:
            self._events.put(("scan_error", str(exc)))

    def _populate_cameras(self, cams):
        try:
            selected = self.cam_index_var.get()
        except tk.TclError:
            selected = 0
        self._cams = cams
        self._cam_label_to_index = {c["label"]: c["index"] for c in cams}
        self.cam_dropdown.configure(values=[c["label"] for c in cams])
        if cams:
            camera = next((c for c in cams if c["index"] == selected), cams[0])
            self._select_camera(camera["label"])
            self.status.config(text=f"Status: Found {len(cams)} camera(s).")
        else:
            self.cam_choice_var.set("(no cameras found)")
            self.status.config(text="Status: No cameras found. Check camera permissions or close other apps.")
        self._set_buttons(False)

    def _select_camera(self, label):
        self.cam_choice_var.set(label)
        self.cam_index_var.set(self._cam_label_to_index[label])

    def _selected_camera_index(self) -> int:
        # The editable index is authoritative; dropdown choices update it too.
        return int(self.cam_index_var.get())

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
    def _build_ui(self):
        self._focus_preview = False
        self._fullscreen = False
        self._preview_size = None
        self._about = None
        self.root.title("PoseFly")
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(1500, screen_w - 60)}x{min(960, screen_h - 80)}")
        self.root.minsize(min(920, screen_w - 60), min(640, screen_h - 80))
        self.root.configure(background=BG)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self._configure_theme()

        header = self._header = ttk.Frame(self.root, padding=(18, 12, 18, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        brand = self._brand = ttk.Frame(header)
        brand.grid(row=0, column=0, sticky="w")
        if hasattr(self, '_icon_ref'):
            ttk.Label(brand, image=self._icon_ref).pack(side="left", padx=(0, 10))
        ttk.Label(brand, text="PoseFly", style="Brand.TLabel").pack(side="left")
        ttk.Label(brand, text="DRONE VISION", style="Muted.TLabel").pack(side="left", padx=16)
        self.btn_about = ttk.Button(header, text="About PoseFly", command=self.show_about)
        self.btn_about.grid(row=0, column=2, sticky="e")
        # A separate action row leaves room for every label, including running
        # and full-screen states, without shrinking any button.
        actions = self._toolbar_actions = ttk.Frame(header)
        actions.grid(row=1, column=0, columnspan=3, sticky="w", pady=(10, 0))
        self.btn_start = ttk.Button(actions, text="Start camera", command=self.start, style="Primary.TButton", width=16)
        self.btn_stop = ttk.Button(actions, text="Stop camera", command=self.stop, state="disabled", style="Stop.TButton", width=14)
        self.btn_focus = ttk.Button(actions, text="Focus preview", command=self.toggle_focus_preview, width=15)
        self.btn_fullscreen = ttk.Button(actions, text="Full screen", command=self.toggle_fullscreen, width=16)
        self._toolbar_buttons = (self.btn_start, self.btn_stop, self.btn_focus, self.btn_fullscreen)
        for column, button in enumerate(self._toolbar_buttons):
            button.grid(row=0, column=column, padx=(0, 12), sticky="w")
        header.bind("<Configure>", self._layout_toolbar)

        self.body = ttk.Frame(self.root, padding=(16, 0, 16, 0))
        self.body.grid(row=1, column=0, sticky="nsew")
        self.body.rowconfigure(0, weight=1)
        self.body.columnconfigure(0, weight=1)
        video = ttk.Frame(self.body)
        video.grid(row=0, column=0, sticky="nsew")
        video.rowconfigure(1, weight=1)
        video.columnconfigure(0, weight=1)
        video_header = ttk.Frame(video, padding=(2, 4, 2, 8))
        video_header.grid(row=0, column=0, sticky="ew")
        video_header.columnconfigure(1, weight=1)
        self.preview_title = ttk.Label(video_header, text="Live preview", style="Heading.TLabel")
        self.preview_title.grid(row=0, column=0, sticky="w")
        self.performance = ttk.Label(video_header, text="Camera: — | Detection: —", style="Muted.TLabel", anchor="e")
        self.performance.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.preview = tk.Canvas(video, background="#060c16", highlightthickness=1,
                                 highlightbackground=BORDER, width=1, height=1)
        self.preview.grid(row=1, column=0, sticky="nsew")
        self._preview_item = self.preview.create_image(0, 0, anchor="center")
        self._preview_placeholder = self.preview.create_text(
            0, 0, text="Your camera preview\n\nChoose a camera, then select Start camera.",
            fill=MUTED, font=("DejaVu Sans", 14), justify="center", anchor="center")
        self.preview.bind("<Configure>", self._on_preview_resize)
        self.preview_hint = ttk.Label(video, text="Amber: older detection  •  F11: full screen  •  Esc: return",
                                      style="Muted.TLabel", padding=(2, 6))
        self.preview_hint.grid(row=2, column=0, sticky="w")

        self.sidebar = ttk.Frame(self.body, width=370)
        self.sidebar.grid(row=0, column=1, sticky="nsew", padx=(16, 0))
        self.sidebar.grid_propagate(False)
        self.sidebar.columnconfigure(0, weight=1)
        self.sidebar.rowconfigure(0, weight=1)
        self.settings_tabs = ttk.Notebook(self.sidebar)
        self.settings_tabs.grid(row=0, column=0, sticky="nsew")
        self._scroll_canvases = []
        self._settings_layouts = []
        camera_tab, camera_actions = self._add_settings_page("Camera")
        detection_tab, detection_actions = self._add_settings_page("Detection")
        # This page has no footer actions; do not reserve blank footer space
        # that could push its model switches below the visible viewport.
        detection_actions.grid_remove()
        recording_tab, recording_actions = self._add_settings_page("Recording")
        logging_tab, logging_actions = self._add_settings_page("Logs")

        ttk.Label(camera_tab, text="Camera source", style="Heading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.cam_dropdown = ttk.Combobox(camera_tab, textvariable=self.cam_choice_var, state="readonly", width=25)
        self.cam_dropdown.grid(row=1, column=0, sticky="ew")
        self.cam_dropdown.bind("<<ComboboxSelected>>", lambda _event: self._select_camera(self.cam_choice_var.get()))
        source_row = ttk.Frame(camera_tab)
        source_row.grid(row=2, column=0, sticky="ew", pady=(10, 12))
        ttk.Label(source_row, text="Camera index").pack(side="left", padx=(0, 12))
        self.cam_index_entry = ttk.Spinbox(source_row, from_=0, to=999, width=5, textvariable=self.cam_index_var)
        self.cam_index_entry.pack(side="left")
        rate_row = ttk.Frame(camera_actions)
        rate_row.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Label(rate_row, text="Live FPS target").pack(side="left", padx=(0, 12))
        self.live_fps_entry = ttk.Spinbox(rate_row, from_=1, to=240, increment=1, width=7,
                                         textvariable=self.live_fps_var, command=self.apply_live_fps)
        self.live_fps_entry.pack(side="left")
        self.live_fps_entry.bind("<Return>", lambda _event: self.apply_live_fps())
        self.live_fps_entry.bind("<FocusOut>", lambda _event: self.apply_live_fps())
        self.live_fps_var.trace_add("write", self._schedule_apply_live_fps)
        self.live_fps_status = ttk.Label(camera_actions, text="1–240 FPS · applies live; camera limits apply.",
                                         style="Muted.TLabel", wraplength=300)
        self.live_fps_status.grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.btn_refresh = ttk.Button(camera_actions, text="Scan cameras", command=self.refresh_cameras)
        self.btn_refresh.grid(row=2, column=0, sticky="ew", padx=(0, 8))
        self.btn_apply_rs = ttk.Button(camera_actions, text="Apply exposure", command=self.apply_rs, state="disabled")
        self.btn_apply_rs.grid(row=2, column=1, sticky="ew")
        camera_actions.columnconfigure((0, 1), weight=1, uniform="camera_actions")
        self.chk_dshow = ttk.Checkbutton(camera_tab, text="Use DirectShow", variable=self.use_dshow_var)
        if platform.system() == "Windows":
            self.chk_dshow.grid(row=3, column=0, sticky="w")
        else:
            self.use_dshow_var.set(False)
            self.chk_dshow.configure(state="disabled")
        exposure = ttk.LabelFrame(camera_tab, text="Exposure", padding=10)
        exposure.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        exposure.columnconfigure((0, 1), weight=1)
        ttk.Checkbutton(exposure, text="Manual camera exposure", variable=self.use_rs_var,
                        command=self.apply_rs).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(exposure, text="ISO").grid(row=1, column=0, sticky="w", pady=(6, 4))
        ttk.Label(exposure, text="Shutter (1/s)").grid(row=1, column=1, sticky="w", pady=(6, 4))
        self.iso_entry = ttk.Entry(exposure, textvariable=self.iso_var, width=9)
        self.iso_entry.grid(row=2, column=0, sticky="ew", padx=(0, 8))
        self.shutter_entry = ttk.Entry(exposure, textvariable=self.shutter_var, width=9)
        self.shutter_entry.grid(row=2, column=1, sticky="ew")
        for entry in (self.iso_entry, self.shutter_entry):
            entry.bind("<KeyRelease>", self._schedule_apply_rs)
            entry.bind("<FocusOut>", self._schedule_apply_rs)
        self.camera_settings_status = ttk.Label(exposure, text="1000 = 1/1000 s exposure.\nISO requires camera support.",
                                                style="Muted.TLabel", wraplength=280, justify="left")
        self.camera_settings_status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(12, 2))

        ttk.Label(detection_tab, text="Drone model", style="Heading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Combobox(detection_tab, values=list(DRONE_MODELS), textvariable=self.drone_model_var,
                     state="readonly", width=25).grid(row=1, column=0, sticky="ew")
        ttk.Label(detection_tab, text="Run models", style="Heading.TLabel").grid(row=2, column=0, sticky="w", pady=(16, 4))
        switches = self._model_switches = ttk.Frame(detection_tab)
        switches.grid(row=3, column=0, sticky="ew")
        switches.columnconfigure((0, 1), weight=1)
        self.feature_switches = []
        for index, (title, variable) in enumerate((("Drone detection", self.tog_drone), ("Angle", self.tog_angle),
                                                ("Distance", self.tog_distance), ("LED ID", self.tog_led),
                                                ("Speed", self.tog_speed))):
            switch = ttk.Checkbutton(switches, text=title, variable=variable)
            row, column = (0, 0) if index == 0 else (1 + (index - 1) // 2, (index - 1) % 2)
            switch.grid(row=row, column=column, columnspan=2 if index == 0 else 1, sticky="w")
            self.feature_switches.append(switch)
        switches.bind("<Configure>", lambda _event: self._fit_sidebar())
        ttk.Label(detection_tab, text="All features stay visible. N/A means no estimate is available, including when a model is off or still running.",
                  style="Muted.TLabel", wraplength=300).grid(row=4, column=0, sticky="w", pady=(10, 0))

        ttk.Label(recording_tab, text="Recording", style="Heading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 6))
        self.save_check = ttk.Checkbutton(recording_tab, text="Save analysis video", variable=self.save_var)
        self.save_check.grid(row=1, column=0, sticky="w")
        ttk.Label(recording_tab, text="Recording playback FPS").grid(row=2, column=0, sticky="w", pady=(10, 4))
        self.fps_entry = ttk.Entry(recording_tab, textvariable=self.fps_var, width=9)
        self.fps_entry.grid(row=3, column=0, sticky="w")
        ttk.Label(recording_tab, text="Output file").grid(row=4, column=0, sticky="w", pady=(16, 4))
        self.out_entry = ttk.Entry(recording_tab, textvariable=self.out_path_var, width=25)
        self.out_entry.grid(row=5, column=0, sticky="ew")
        self.btn_browse = ttk.Button(recording_actions, text="Choose output file…", command=self.choose_output)
        self.btn_browse.grid(row=0, column=0, sticky="ew")
        recording_actions.columnconfigure(0, weight=1)
        ttk.Label(recording_tab, text="Set these before starting. Recordings contain the analyzed frames and their labels.",
                  style="Muted.TLabel", wraplength=300).grid(row=6, column=0, sticky="w", pady=(14, 0))

        ttk.Label(logging_tab, text="Detection log", style="Heading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Label(logging_tab, text="CSV file").grid(row=1, column=0, sticky="w", pady=(6, 4))
        self.log_path_entry = ttk.Entry(logging_tab, textvariable=self.log_path_var, width=25)
        self.log_path_entry.grid(row=2, column=0, sticky="ew")
        ttk.Label(logging_tab, text="Save timestamps, drone boxes, confidence, angle, distance, LED ID, and speed. Missing estimates are N/A.",
                  style="Muted.TLabel", wraplength=300).grid(row=3, column=0, sticky="w", pady=(14, 8))
        ttk.Label(logging_tab, text="Start or stop logging at any time, even without video recording. Existing log rows are kept. Stopping the camera finishes the log.",
                  style="Muted.TLabel", wraplength=300).grid(row=4, column=0, sticky="w", pady=(0, 8))
        self.log_error_detail = ttk.Label(logging_tab, text="", style="Error.TLabel", wraplength=300)
        self.log_error_detail.grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.log_status = ttk.Label(logging_actions, text="Logging is off.", style="Muted.TLabel", wraplength=300)
        self.log_status.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))
        self.btn_log_browse = ttk.Button(logging_actions, text="Choose log file…", command=self.choose_log_output)
        self.btn_log_browse.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.btn_logging = ttk.Button(logging_actions, text="Start logging", command=self.toggle_logging, style="Primary.TButton")
        self.btn_logging.grid(row=1, column=1, sticky="ew")
        logging_actions.columnconfigure((0, 1), weight=1, uniform="log_actions")

        # Results occupy the full window width and remain visible in focus mode.
        results = self._results_panel = ttk.LabelFrame(self.root, text="Drone features", padding=(10, 6))
        results.grid(row=2, column=0, sticky="ew", padx=16, pady=(8, 0))
        results.columnconfigure(0, weight=1)
        self.feature_summary = ttk.Label(results, text="Waiting for drone detection. N/A = no estimate available.",
                                         style="Muted.TLabel", anchor="w")
        self.feature_summary.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        columns = ("drone", "confidence", "angle", "distance", "led", "speed")
        self.feature_table = ttk.Treeview(results, columns=columns, show="headings", height=2, selectmode="browse")
        self.feature_table.grid(row=1, column=0, sticky="ew")
        heading_font = tkfont.Font(root=self.root, font=ttk.Style(self.root).lookup("Treeview.Heading", "font"))
        self._results_min_width = 60
        for name, title in zip(columns, ("Drone", "Confidence", "Angle", "Distance", "LED ID", "Speed")):
            self.feature_table.heading(name, text=title)
            minimum = max(95, heading_font.measure(title) + 40)
            self._results_min_width += minimum
            self.feature_table.column(name, width=max(130, minimum), minwidth=minimum, anchor="center", stretch=True)
        scrollbar = ttk.Scrollbar(results, command=self.feature_table.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.feature_table.configure(yscrollcommand=scrollbar.set)
        self._result_rows = None
        self._update_feature_rows([])

        self.status = ttk.Label(self.root, text="Status: Idle", style="Muted.TLabel", anchor="w", padding=(18, 8))
        self.status.grid(row=3, column=0, sticky="ew")
        self.root.bind("<F11>", lambda _event: self.toggle_fullscreen())
        self.root.bind("<Escape>", self._exit_expanded_view)
        self.root.bind("<F1>", lambda _event: self.show_about())
        for event_name in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.root.bind(event_name, self._scroll_sidebar, add="+")
        self.root.after_idle(self._fit_sidebar)

    def _configure_theme(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("DejaVu Sans", 10), background=BG, foreground=TEXT,
                        bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER, troughcolor=BG,
                        selectbackground="#245ab8", selectforeground="#ffffff")
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Muted.TLabel", foreground=MUTED, font=("DejaVu Sans", 9))
        style.configure("Error.TLabel", foreground="#ffb6bf", font=("DejaVu Sans", 9))
        style.configure("Brand.TLabel", font=("DejaVu Sans", 22, "bold"))
        style.configure("Heading.TLabel", font=("DejaVu Sans", 11, "bold"))
        style.configure("TButton", padding=(14, 10), background=FIELD, foreground=TEXT,
                        bordercolor=BORDER, focusthickness=2, focuscolor="#78adff")
        style.map("TButton", background=[("disabled", "#152033"), ("pressed", "#29466a"), ("active", "#223854")],
                  foreground=[("disabled", "#7f91aa")], bordercolor=[("focus", "#78adff")])
        style.configure("Primary.TButton", background="#2563eb", foreground="#ffffff", font=("DejaVu Sans", 10, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#152033"), ("pressed", "#1e40af"), ("active", "#3478fa")],
                  foreground=[("disabled", "#8ea1ba"), ("!disabled", "#ffffff")])
        style.configure("Stop.TButton", background="#502332", foreground="#ffb6bf", font=("DejaVu Sans", 10, "bold"))
        style.map("Stop.TButton", background=[("disabled", "#152033"), ("pressed", "#6e2538"), ("active", "#703046")],
                  foreground=[("disabled", "#7f91aa"), ("!disabled", "#ffd4da")])
        style.configure("TCheckbutton", background=BG, foreground=TEXT, padding=(0, 6),
                        indicatorbackground=FIELD, indicatorforeground="#ffffff")
        style.map("TCheckbutton", background=[("active", PANEL)], foreground=[("disabled", "#7f91aa")],
                  indicatorbackground=[("selected", "#2563eb"), ("!selected", FIELD)])
        for name in ("TEntry", "TSpinbox", "TCombobox"):
            style.configure(name, padding=7, fieldbackground=FIELD, foreground=TEXT, insertcolor=TEXT,
                            background=FIELD, arrowcolor=TEXT)
            style.map(name, fieldbackground=[("disabled", "#152033"), ("readonly", FIELD)],
                      foreground=[("disabled", "#8ea1ba"), ("readonly", TEXT)],
                      selectbackground=[("!focus", FIELD)], selectforeground=[("!focus", TEXT)])
        style.configure("TLabelframe", background=BG, bordercolor=BORDER)
        style.configure("TLabelframe.Label", background=BG, foreground=TEXT, font=("DejaVu Sans", 10, "bold"))
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 10), background=FIELD, foreground=MUTED)
        style.map("TNotebook.Tab", background=[("selected", "#223854"), ("active", "#1a2e49")],
                  foreground=[("selected", "#ffffff")])
        style.configure("Vertical.TScrollbar", background=BORDER, arrowcolor=TEXT, troughcolor=PANEL,
                        bordercolor=PANEL, lightcolor=BORDER, darkcolor=BORDER)
        style.map("Vertical.TScrollbar", background=[("active", "#526c90"), ("pressed", "#526c90")])
        result_font = tkfont.Font(root=self.root, font=style.lookup("Treeview", "font"))
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        rowheight=max(30, result_font.metrics("linespace") + 8), bordercolor=BORDER)
        style.configure("Treeview.Heading", background=FIELD, foreground=TEXT, padding=(8, 7),
                        font=("DejaVu Sans", 10, "bold"))
        style.map("Treeview", background=[("selected", "#24466e")], foreground=[("selected", "#ffffff")])
        style.map("Treeview.Heading", background=[("active", "#223854")])
        # Tk's combobox popdown and file chooser contain classic widgets.
        for option, value in (("background", BG), ("foreground", TEXT), ("selectBackground", "#245ab8"),
                              ("selectForeground", "#ffffff"), ("insertBackground", TEXT),
                              ("troughColor", PANEL), ("activeBackground", FIELD), ("activeForeground", TEXT)):
            self.root.option_add(f"*{option}", value)

    def _layout_toolbar(self, event):
        button_widths = [button.winfo_reqwidth() + 12 for button in self._toolbar_buttons]
        needed = self._brand.winfo_reqwidth() + sum(button_widths) + self.btn_about.winfo_reqwidth() + 60
        if event.width >= needed:
            self._toolbar_actions.grid(row=0, column=1, columnspan=1, pady=0)
            columns = len(self._toolbar_buttons)
        else:
            self._toolbar_actions.grid(row=1, column=0, columnspan=3, pady=(10, 0))
            available = max(1, event.width - 36)
            # Account for the widest button in each grid column, not only the
            # sum of one row; enlarged fonts can require two or more rows.
            for columns in range(len(self._toolbar_buttons), 0, -1):
                if sum(max(button_widths[column::columns]) for column in range(columns)) <= available:
                    break
        for index, button in enumerate(self._toolbar_buttons):
            button.grid(row=index // columns, column=index % columns,
                        pady=(0, 8 if index // columns < (len(self._toolbar_buttons) - 1) // columns else 0))
        self._fit_sidebar()

    def _fit_sidebar(self):
        # Canvas children do not contribute their natural size to the notebook.
        # Measure them explicitly: fonts scale with DPI, fixed pixel widths do not.
        width = max(370, self.settings_tabs.winfo_reqwidth() + 8)
        for page, scrollbar, actions in self._settings_layouts:
            width = max(width, page.winfo_reqwidth() + scrollbar.winfo_reqwidth() + 8,
                        actions.winfo_reqwidth() + 8)
        if int(self.sidebar.cget("width")) != width:
            self.sidebar.configure(width=width)
        screen_w, screen_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        selected_page = self.settings_tabs.nametowidget(self.settings_tabs.select())
        tab_height = max(40, self.settings_tabs.winfo_height() - selected_page.winfo_height())
        controls_height = self._model_switches.winfo_y() + self._model_switches.winfo_reqheight() + 12
        minimum_height = max(640, self._header.winfo_reqheight() + self._results_panel.winfo_reqheight()
                             + self.status.winfo_reqheight() + tab_height + controls_height + 8)
        self.root.minsize(min(max(920, width + 320, self._results_min_width), screen_w - 60),
                          min(minimum_height, screen_h - 80))

    def _add_settings_page(self, title):
        shell = ttk.Frame(self.settings_tabs)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(0, weight=1)
        canvas = tk.Canvas(shell, background=BG, highlightthickness=0, width=1, height=1)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(shell, command=canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)
        page = ttk.Frame(canvas, padding=12)
        page.columnconfigure(0, weight=1)
        item = canvas.create_window(0, 0, window=page, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(item, width=event.width))
        def resize_page(_event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            self._fit_sidebar()
        page.bind("<Configure>", resize_page)
        # Actions are outside the scrolling content and always remain in view.
        actions = ttk.Frame(shell, padding=(10, 8))
        actions.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._settings_layouts.append((page, scrollbar, actions))
        self._scroll_canvases.append(canvas)
        self.settings_tabs.add(shell, text=title)
        return page, actions

    def _update_feature_rows(self, rows):
        rows = rows or [("N/A",) * 6]
        if rows == self._result_rows:
            return
        self._result_rows = rows
        # Update in place so selection and scroll position survive live results.
        for index, values in enumerate(rows):
            iid = str(index)
            if self.feature_table.exists(iid):
                self.feature_table.item(iid, values=values)
            else:
                self.feature_table.insert("", "end", iid=iid, values=values)
        for iid in self.feature_table.get_children()[len(rows):]:
            self.feature_table.delete(iid)

    def show_about(self):
        if self._about is not None and self._about.winfo_exists():
            self._about.lift()
            self._about.focus_set()
            return "break"
        dialog = self._about = tk.Toplevel(self.root, class_="Posefly")
        dialog.title("About PoseFly")
        dialog.configure(background=BG)
        dialog.transient(self.root)
        dialog.resizable(False, False)
        if hasattr(self, '_icon_ref'):
            dialog.iconphoto(False, self._icon_ref)
        content = ttk.Frame(dialog, padding=24, relief="solid", borderwidth=1)
        content.pack(fill="both", expand=True)
        if hasattr(self, '_icon_ref'):
            ttk.Label(content, image=self._icon_ref).pack(pady=(0, 8))
        ttk.Label(content, text="PoseFly Research Project", style="Brand.TLabel").pack()
        ttk.Label(content, text="Version: 1.0\nDeveloper: Deniz K. Acikbas\nProgramming Language: Python",
                  justify="center").pack(pady=(8, 0))
        controls = ttk.Frame(content)
        controls.pack(pady=(22, 0))
        ttk.Button(controls, text="Project on GitHub", command=lambda: webbrowser.open(PROJECT_URL)).pack(side="left", padx=(0, 12))
        close = ttk.Button(controls, text="Close", command=dialog.destroy, style="Primary.TButton")
        close.pack(side="left")
        dialog.bind("<Escape>", lambda _event: dialog.destroy())
        dialog.update_idletasks()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - dialog.winfo_reqwidth()) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - dialog.winfo_reqheight()) // 2
        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")
        close.focus_set()
        return "break"

    def _scroll_sidebar(self, event):
        widget = event.widget
        if isinstance(widget, (ttk.Combobox, ttk.Spinbox)):
            return
        while widget is not None:
            if widget in self._scroll_canvases:
                if widget.yview() == (0., 1.):
                    return
                units = (-1 if event.num == 4 else 1) if event.num in (4, 5) else (-1 if event.delta > 0 else 1)
                widget.yview_scroll(units, "units")
                return "break"
            widget = getattr(widget, "master", None)

    def _on_preview_resize(self, event):
        self._preview_size = (max(1, event.width - 4), max(1, event.height - 4))
        center = (event.width / 2, event.height / 2)
        self.preview.coords(self._preview_item, *center)
        self.preview.coords(self._preview_placeholder, *center)
        self.preview.itemconfigure(self._preview_placeholder, width=max(100, event.width - 40))
        available = event.width - self.preview_title.winfo_reqwidth() - 20
        if available < 200:
            self.performance.grid(row=1, column=0, columnspan=2, sticky="ew", padx=0)
            self.performance.configure(wraplength=max(1, event.width - 8), anchor="w")
        else:
            self.performance.grid(row=0, column=1, columnspan=1, sticky="ew", padx=(12, 0))
            self.performance.configure(wraplength=available, anchor="e")
        self.preview_hint.configure(wraplength=max(180, event.width - 8))
        self.status.configure(wraplength=max(300, self.root.winfo_width() - 36))
        self.feature_summary.configure(wraplength=max(300, self.root.winfo_width() - 72))

    def toggle_focus_preview(self):
        self._focus_preview = not self._focus_preview
        if self._focus_preview:
            self.sidebar.grid_remove()
        else:
            self.sidebar.grid()
        self.btn_focus.configure(text="Show controls" if self._focus_preview else "Focus preview")
        return "break"

    def toggle_fullscreen(self):
        self._fullscreen = not self._fullscreen
        self.root.attributes("-fullscreen", self._fullscreen)
        self.btn_fullscreen.configure(text="Exit full screen" if self._fullscreen else "Full screen")
        return "break"

    def _exit_expanded_view(self, _event=None):
        if self._fullscreen:
            self.toggle_fullscreen()
        if self._focus_preview:
            self.toggle_focus_preview()
        return "break"

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

    def choose_log_output(self):
        path = filedialog.asksaveasfilename(title="Choose detection log", defaultextension=".csv",
                                           filetypes=[("CSV log", "*.csv")], confirmoverwrite=False,
                                           initialfile=Path(self.log_path_var.get()).name)
        if path:
            self.log_path_var.set(path)

    def toggle_logging(self):
        if self._log_state in ("starting", "active", "stopping"):
            self.stop_logging()
            return
        path = self.log_path_var.get().strip()
        if not path or Path(path).suffix.lower() != ".csv":
            self._set_log_state("error", 0, "Choose a .csv file for the detection log.")
            return
        self._log_generation += 1
        generation = self._log_generation
        self._set_log_state("starting", 0, "")
        self._log_writer = ResultLogWriter(path, lambda state, rows, detail:
                                           self._events.put(("logging", (generation, state, rows, detail))))
        self._log_writer.start()

    def stop_logging(self):
        if self._log_writer is not None:
            if self._log_state in ("starting", "active"):
                self._set_log_state("stopping", self._log_rows, "")
            self._log_writer.stop()

    def _set_log_state(self, state, rows, detail):
        if state == "active" and self._log_state == "stopping":
            state = "stopping"  # A queued progress update must not undo Stop.
        self._log_state, self._log_rows = state, rows
        busy = state in ("starting", "active", "stopping")
        self.log_path_entry.configure(state="readonly" if busy else "normal")
        self.btn_log_browse.configure(state="disabled" if busy else "normal")
        self.btn_logging.configure(
            text={"starting": "Opening log…", "active": "Stop logging", "stopping": "Finishing log…"}.get(state, "Start logging"),
            state="disabled" if state in ("starting", "stopping") else "normal",
            style="Stop.TButton" if state == "active" else "Primary.TButton")
        text = {"starting": "Opening CSV log…", "active": f"Logging active · {rows} rows saved.",
                "stopping": "Finishing pending log rows…", "stopped": f"Log saved · {rows} rows this session.",
                "error": "Logging failed. Check the file or location."}.get(state, "Logging is off.")
        self.log_status.configure(text=text, style="Error.TLabel" if state == "error" else "Muted.TLabel")
        self.log_error_detail.configure(text=detail if state == "error" else "")

    def _schedule_apply_live_fps(self, *_args):
        if self._fps_after_id:
            self.root.after_cancel(self._fps_after_id)
        self._fps_after_id = self.root.after(200, self.apply_live_fps)

    def apply_live_fps(self):
        if self._fps_after_id:
            self.root.after_cancel(self._fps_after_id)
            self._fps_after_id = None
        try:
            fps = float(self.live_fps_var.get())
            if not math.isfinite(fps) or not 1 <= fps <= 240:
                raise ValueError
        except (ValueError, tk.TclError):
            self.live_fps_status.configure(text=f"Enter 1–240 FPS. Keeping target {self._live_fps:g} FPS.")
            return False
        changed = fps != self._live_fps
        self._live_fps = fps
        self._snapshot_settings()
        if changed:
            # Wake a low-FPS wait immediately; the worker recomputes its deadline.
            self._capture_wakeup.set()
        self.live_fps_status.configure(text=f"Target {fps:g} FPS · applies live; camera limits apply.")
        return True

    # --------------------------------------------------------
    # Rolling shutter handling
    # --------------------------------------------------------
    def _read_rs_inputs(self) -> tuple[int, float]:
        iso = clamp_int(self.iso_var.get(), 50, 6400, 250)
        shz = clamp_float(self.shutter_var.get(), 5.0, 6000.0, 1000.0)
        return iso, shz

    def _schedule_apply_rs(self, _event=None):
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

        self._snapshot_settings()
        self._rs_request += 1
        self.camera_settings_status.config(text=f"Applying ISO {iso}, exposure 1/{shz:g} s…")

    def apply_rs(self):
        self._apply_rs_if_changed(force=True)

    # --------------------------------------------------------
    # Run pipeline
    # --------------------------------------------------------
    def start(self):
        if self._closing or (self._thread and self._thread.is_alive()):
            return
        if not self.apply_live_fps():
            return

        try:
            cam_idx = self._selected_camera_index()
            if cam_idx < 0:
                raise ValueError("Camera index must be nonnegative")
        except (ValueError, tk.TclError):
            messagebox.showwarning("Invalid camera", "Enter a nonnegative camera index.")
            return
        use_dshow = bool(self.use_dshow_var.get()) if platform.system() == "Windows" else False
        fps = clamp_float(self.fps_var.get(), 1.0, 240.0, 60.0)
        out_path = str(self.out_path_var.get()).strip()
        save_video = bool(self.save_var.get())

        if save_video and not out_path:
            messagebox.showwarning("Missing output path", "Choose an output path or disable Save analysis video.")
            return

        iso, shz = self._read_rs_inputs()
        self._rs_iso_cached, self._rs_shz_cached = iso, shz

        self._snapshot_settings()
        self._stop_event.clear()
        self._capture_wakeup.clear()
        with self._frame_lock:
            self._latest_frame = None
            self._display_frame = None
            self._detections_model = None
            self._overlay_revision += 1
            self._detections = []
            self._features = []
            self._features_pending = False
            self._detections_captured_at = self._detections_shape = None
            self._camera_fps = self._analysis_seconds = 0.0
            self._analysis_finished_at = None

        self._running = True
        self._frame_count = 0
        self._last_rs_applied = (None, None)

        self._set_buttons(True)
        self.preview.itemconfigure(self._preview_item, image="")
        self.preview.itemconfigure(self._preview_placeholder, state="normal", text="Starting camera…")
        self.status.config(text="Status: Starting camera…")

        self._thread = threading.Thread(
            target=self._pipeline_loop,
            args=(cam_idx, use_dshow, fps, out_path, save_video),
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._capture_wakeup.set()
        self.stop_logging()
        self._running = False
        self.status.config(text="Status: Stopping...")
        self.btn_stop.config(state="disabled", text="Stopping…")

    def _snapshot_settings(self):
        # Called only on Tk's main thread; the worker reads an immutable snapshot.
        self._settings = (
            {"drone": self.tog_drone.get(), "angle": self.tog_angle.get(),
             "distance": self.tog_distance.get(), "led": self.tog_led.get(),
             "speed": self.tog_speed.get(), "drone_model": self.drone_model_var.get()},
            self.use_rs_var.get(), self._rs_iso_cached, self._rs_shz_cached, self._live_fps,
        )

    def _analysis_loop(self, frames, fps, out_path, save_video, errors):
        try:
            while not self._stop_event.is_set():
                item = frames.get()
                if item is None or self._stop_event.is_set():
                    break
                frame, settings, captured_at, captured_utc = item
                source_shape = frame.shape[:2]
                toggles = settings[0]
                started = time.monotonic()
                def publish_detections(detections):
                    with self._frame_lock:
                        self._detections = list(detections)
                        self._detections_model = toggles.get("drone_model", "Indoor")
                        self._overlay_revision += 1
                        self._features = []
                        self._features_pending = bool(detections)
                        self._detections_captured_at = captured_at
                        self._detections_shape = source_shape
                out = self.backend.process_frame(frame, toggles, on_detections=publish_detections)
                if isinstance(out, np.ndarray):
                    frame = out
                elif isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], np.ndarray):
                    frame = out[1]
                if save_video:
                    try:
                        self.backend.write_frame_if_enabled(frame, True, out_path, fps)
                    except Exception as exc:
                        save_video = False
                        self._set_status_threadsafe(f"Status: Recording disabled: {exc}")
                        self.backend.release_writer()
                finished = time.monotonic()
                with self._frame_lock:
                    detections = self.backend.drone_detections()
                    self._detections = list(detections) if isinstance(detections, (list, tuple)) else []
                    self._detections_model = toggles.get("drone_model", "Indoor")
                    self._overlay_revision += 1
                    features = self.backend.drone_features()
                    self._features = list(features) if isinstance(features, (list, tuple)) else []
                    self._features_pending = False
                    self._detections_captured_at = captured_at
                    self._detections_shape = source_shape
                    self._analysis_seconds = finished - started
                    self._analysis_finished_at = finished
                    logged_detections, logged_features = self._detections, self._features
                logger = self._log_writer
                if logger is not None:
                    logger.submit(logged_detections, logged_features, captured_utc,
                                  toggles.get("drone_model", "Indoor"), toggles.get("drone", True))
        except Exception as exc:
            errors.append(str(exc))
            traceback.print_exc()
            self._stop_event.set()
            self._capture_wakeup.set()

    def _wait_for_next_frame(self, started):
        while True:
            # Clear before reading settings/stop state so a concurrent change
            # either appears in this read or wakes the subsequent wait.
            self._capture_wakeup.clear()
            if self._stop_event.is_set():
                return
            delay = 1.0 / self._settings[4] - (time.monotonic() - started)
            if delay <= 0 or not self._capture_wakeup.wait(delay):
                return

    def _pipeline_loop(self, cam_idx, use_dshow, fps, out_path, save_video):
        error = None
        frames = LatestFrame()
        analysis_thread = None
        analysis_errors = []
        try:
            self.backend.open_camera(camera_index=cam_idx, use_dshow=use_dshow)
            self.backend.apply_camera_settings_led_id()
            analysis_thread = threading.Thread(
                target=self._analysis_loop,
                args=(frames, fps, out_path, save_video, analysis_errors),
                daemon=True,
            )
            analysis_thread.start()
            self._set_status_threadsafe("Status: Live camera running; AI analysis runs separately")
            applied_rs = None
            measured_target = None
            measured_since, frame_count = time.monotonic(), 0
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                settings = self._settings
                _, use_rs, iso, shz, live_fps = settings
                if live_fps != measured_target:
                    measured_since, frame_count = t0, 0
                    measured_target = live_fps
                request = (use_rs, iso, shz, self._rs_request)
                if applied_rs != request:
                    report = self.backend.set_rollingshutter(iso, shz, enabled=use_rs)
                    self._events.put(("camera_settings", str(report)))
                    applied_rs = request
                ok, frame = self.backend.read_frame()
                if not ok or frame is None:
                    raise RuntimeError("Failed to read frame; check the camera connection.")
                if self._stop_event.is_set():
                    break
                # Analysis may annotate its own copy; it cannot block or alter live video.
                frames.put((frame.copy(), settings, time.monotonic(), datetime.now(timezone.utc)))
                self._show_frame_threadsafe(frame)
                frame_count += 1
                elapsed = time.monotonic() - measured_since
                if elapsed >= 0.5:
                    with self._frame_lock:
                        self._camera_fps = frame_count / elapsed
                    measured_since, frame_count = time.monotonic(), 0
                self._wait_for_next_frame(t0)
        except Exception as exc:
            error = str(exc)
            traceback.print_exc()
        finally:
            self._stop_event.set()
            frames.close()
            if analysis_thread is not None:
                analysis_thread.join()
            if analysis_errors:
                error = error or analysis_errors[0]
            # Both workers are finished before capture/writer/log resources are closed.
            for release in (self.backend.release_writer, self.backend.release_camera):
                try:
                    release()
                except Exception as exc:
                    error = error or str(exc)
            if self._log_writer is not None:
                self._log_writer.stop()
            self._running = False
            self._events.put(("finished", error))

    def _set_status_threadsafe(self, text):
        self._events.put(("status", text))

    def _set_buttons(self, running):
        self.btn_start.config(state="disabled" if running else "normal", text="Camera running" if running else "Start camera")
        self.btn_stop.config(state="normal" if running else "disabled", text="Stop camera")
        self.btn_apply_rs.config(state="normal" if running else "disabled")
        self.btn_refresh.config(state="disabled" if running else "normal", text="Scan cameras")
        self.cam_dropdown.config(state="disabled" if running else "readonly")
        for widget in (self.cam_index_entry, self.save_check, self.fps_entry, self.out_entry, self.btn_browse):
            widget.config(state="disabled" if running else "normal")
        if platform.system() == "Windows":
            self.chk_dshow.config(state="disabled" if running else "normal")

    def _show_frame_threadsafe(self, frame_bgr):
        # One pending frame bounds memory and avoids a backlog of stale previews.
        with self._frame_lock:
            self._latest_frame = frame_bgr

    def _ui_tick(self):
        if self._closing:
            return
        tick_started = time.monotonic()
        self._snapshot_settings()
        while not self._events.empty():
            kind, value = self._events.get_nowait()
            if kind == "status":
                self.status.config(text=value)
            elif kind == "camera_settings":
                self.camera_settings_status.config(text=value)
            elif kind == "logging":
                generation, state, rows, detail = value
                if generation == self._log_generation:
                    self._set_log_state(state, rows, detail)
            elif kind == "cameras":
                self._populate_cameras(value)
            elif kind == "scan_error":
                self._set_buttons(False)
                self.status.config(text=f"Status: Camera scan error: {value}")
            elif kind == "finished":
                # Do not enable another session until the old worker has exited.
                if self._thread and self._thread.is_alive():
                    self._events.put((kind, value))
                    break
                self._set_buttons(False)
                error = value
                if error:
                    self.status.config(text=f"Status: ERROR {error}")
                else:
                    self.status.config(text="Status: Stopped")
        with self._frame_lock:
            frame, self._latest_frame = self._latest_frame, None
            detections = self._detections
            detections_model = self._detections_model
            revision = self._overlay_revision
            features = self._features
            features_pending = self._features_pending
            captured_at = self._detections_captured_at
            source_shape = self._detections_shape
            camera_fps = self._camera_fps
            analysis_seconds = self._analysis_seconds
            analyzed_at = self._analysis_finished_at
        # Hide disabled features immediately, including while an old inference completes.
        features = [{name: value for name, value in item.items() if self._settings[0].get(name)}
                    for item in features]
        now = time.monotonic()
        if detections_model is not None and detections_model != self.drone_model_var.get():
            detections, features, captured_at = [], [], None
        new_frame = frame is not None
        if new_frame:
            self._display_frame = frame
        controls = tuple(self._settings[0].items())
        # Render new results even if the camera has not delivered another frame.
        # Refresh an old result's visible age twice per second without doing idle work.
        age_tick = int(now * 2) if detections else 0
        overlay_key = (revision, controls, age_tick, self._preview_size)
        if self._display_frame is not None and (new_frame or overlay_key != self._displayed_overlay):
            shown = self._display_frame
            if self.tog_drone.get():
                shown = draw_recent_drones(shown, detections, captured_at, source_shape, now, features)
            target_size = self._preview_size
            if target_size is None or min(target_size) < 2:
                target_size = (shown.shape[1], shown.shape[0])
            self._tk_img = self._preview_image(shown, target_size)
            self.preview.itemconfigure(self._preview_item, image=self._tk_img)
            self.preview.itemconfigure(self._preview_placeholder, state="hidden")
            self._displayed_overlay = overlay_key
        if analyzed_at is None:
            analysis_text = "waiting for first result"
        else:
            rate = 1.0 / max(analysis_seconds, 0.000001)
            age = time.monotonic() - analyzed_at
            analysis_text = f"{rate:.1f} FPS | last result {age:.1f}s ago"
        self.performance.config(text=f"Camera: {camera_fps:.1f} FPS (target {self._live_fps:g}) | Detection: {analysis_text}")
        rows = []
        if not self.tog_drone.get():
            summary = "Drone detection is off; feature models are paused."
        elif not detections:
            summary = ("Waiting for drone detection…" if captured_at is None
                       else "No drone detected; feature models are waiting.")
        else:
            age = max(0., time.monotonic() - captured_at)
            state = " — estimating selected features…" if features_pending else ""
            summary = f"Latest detected drones — source frame {age:.1f}s ago{state}"
            for index, detection in enumerate(detections):
                values = feature_values(features[index] if index < len(features) else {})
                rows.append((f"Drone {index + 1}", f"{detection[4]:.0%}", *values))
        self._update_feature_rows(rows)
        log_note = ""
        if self._log_state in ("starting", "active", "stopping"):
            log_note = f"  •  Logging: {self._log_state} ({self._log_rows} rows)"
        elif self._log_state == "error":
            log_note = "  •  Log error — see Logs"
        self.feature_summary.config(text=summary + "  •  N/A = no estimate available." + log_note)
        # Keep controls/results responsive at low capture rates and allow the
        # preview to refresh above 30 FPS when the camera and machine can do so.
        interval = 1.0 / max(30.0, self._live_fps)
        delay_ms = max(1, math.ceil(1000 * (interval - (time.monotonic() - tick_started))))
        self._ui_after_id = self.root.after(delay_ms, self._ui_tick)

    @staticmethod
    def _preview_image(frame, size):
        # Resize before color conversion to keep large camera frames off the UI hot path.
        h, w = frame.shape[:2]
        scale = min(size[0] / w, size[1] / h)
        if abs(scale - 1.0) > 0.001:
            frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))),
                               interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        return ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

    def on_close(self):
        if self._closing:
            return
        self._closing = True
        if self._ui_after_id:
            self.root.after_cancel(self._ui_after_id)
            self._ui_after_id = None
        self.stop()
        self.btn_start.config(state="disabled")
        self.btn_refresh.config(state="disabled")
        if self._rs_after_id:
            self.root.after_cancel(self._rs_after_id)
            self._rs_after_id = None
        if self._fps_after_id:
            self.root.after_cancel(self._fps_after_id)
            self._fps_after_id = None
        self._wait_for_shutdown()

    def _wait_for_shutdown(self):
        if ((self._thread and self._thread.is_alive())
                or (self._log_writer and self._log_writer.is_alive())):
            self.root.after(50, self._wait_for_shutdown)
        else:
            self.root.destroy()


def main():
    # Tk title-cases the application class; match StartupWMClass exactly.
    root = tk.Tk(className="Posefly")
    app = PoseFlyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
