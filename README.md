# PoseFly — Webcam Inference (AI-enhanced, Camera-based Drone-to-Drone Communication System)

This repository contains a **webcam-based inference demo** inspired by **PoseFly**, a **4-in-1 Optical Camera Communication (OCC)** approach for swarming drones that integrates:
1) **Drone identification** (active optical label via rolling-shutter OOK),
2) **On-site localization / pose parsing** (distance, angle, speed estimation),
3) **Quick-link communication** (LED-to-camera burst data),
4) **Lighting** (LEDs as illumination).

PoseFly uses rolling-shutter cameras and simple LED hardware to support **drone-to-drone interactions** without relying on congested RF-only centralized control.

---

## What this codebase does

In this implementation, we run a **webcam inference pipeline** that can optionally enable/disable:
- 🛸 **Drone Detection**
- 🧭 **Angle Estimation**
- 📏 **Distance Estimation**
- 💡 **LED ID / Optical Label ID**
- 🎞️ **Optional video saving**

The original PoseFly paper reports strong prototype results (e.g., near 100% distance estimation accuracy up to 20 m, identification up to 12 m, angle/speed within 4 m, and ~5 Kbps quick-link throughput on prototypes).

---

## GNU/Linux setup

Use the native Python/Tk desktop interface on GNU/Linux. The C# WinForms
interface is Windows-only. Python **3.10 or newer** is required.

On Debian/Ubuntu, install the system GUI and video prerequisites:

```sh
sudo apt install python3 python3-venv python3-tk libgl1 libglib2.0-0 ffmpeg
```

From the repository root, create an isolated Python environment:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r "PoseFly Desktop App/PoseFly.WinForms/PyBackend/requirements.txt"
```

The dependencies include PyTorch through Ultralytics. CPU inference works without
CUDA. For an NVIDIA installation, choose a PyTorch build compatible with your
driver before installing the requirements. The requirements use Ultralytics 8.4.56 or newer, tested with the bundled YOLO26
models; older YOLO-only environments may not load them.

Launch:

```sh
./"PoseFly Desktop App/PoseFly.WinForms/PyBackend/RUN_POSEFLY_LINUX.sh"
```

For a clickable launcher **inside the project folder**, double-click **PoseFly**
(the executable without an extension). Some GNOME file managers open `.desktop`
files in a text editor inside ordinary folders; use the executable there.
The included executable is built for this machine's Linux architecture. Rebuild
it after cloning onto a different architecture, or if it is missing:

```sh
cc -O2 -Wall -Wextra -Werror linux/launcher.c -o PoseFly
```

The executable resolves the project relative to its own location, so it also
works after moving the whole project directory. `PoseFly.desktop` is retained
for desktop environments that support desktop entries; its paths are specific
to this installation.

The launcher registers `posefly.desktop` and the bundled icon in your user's
application/icon directories under `${XDG_DATA_HOME:-$HOME/.local/share}`.
The window class and desktop entry both identify **PoseFly**, allowing Linux
taskbars to match the app to its icon. Registration is refreshed after a move;
it can also be run manually with `python3 linux/install_desktop.py`.
No administrator access is needed. This follows the
[desktop entry specification](https://specifications.freedesktop.org/desktop-entry/latest-single/)
for `Icon` and `StartupWMClass`.

The launcher works from any directory. It chooses `POSEFLY_PYTHON` when set,
then an activated virtual environment, then the repository's `.venv`, then
`python3`. For example, `POSEFLY_PYTHON=/path/to/python` selects another installation.
Run in a graphical desktop session; Tk needs X11 or XWayland (`DISPLAY`).

## Using the Linux interface

- The interface uses a dark theme, including settings, results, and the About
  dialog. **About PoseFly** or **F1** opens the project information dialog.
  Action buttons wrap onto a separate toolbar row on smaller windows, retaining
  their full size. Camera and recording actions stay outside scrolling settings.
  The sidebar measures the actual tab labels and controls at your display's text
  scale. Larger fonts expand the sidebar, and toolbar buttons wrap into additional
  rows when needed. Result rows and column headings also fit the scaled text.
- The preview expands with the window and keeps the full image's proportions.
  **Focus preview** hides the settings sidebar for more viewing space.
  **Full screen** or **F11** fills the screen; **Esc** restores the controls and
  exits full screen. Start and Stop stay accessible in both viewing modes.
  Camera, Detection, Recording, and Logs settings are grouped in sidebar tabs.
  Settings scroll when space is limited. The full-width feature table stays
  visible below the preview, including in Focus preview mode.
- Camera discovery checks existing `/dev/video*` devices using V4L2, including
  sparse/high device numbers. It runs in the background. Select a camera or
  enter its numeric index manually, then press **Start camera**. Camera source
  and recording settings are locked during capture; stop to change them.
- The desktop app shows the **full camera view with drone detections and features**.
  **Drone model: Indoor** is the default for the included illuminated-drone
  dataset. **Outdoor** selects the bundled standard YOLO26 outdoor checkpoint.
  The previous custom-attention drone checkpoint is no longer the default.
  Switching profiles loads each checkpoint once and reuses it.
  After detecting a drone, enabled angle, distance, LED-ID and speed models run
  on that drone's unmodified crop. They do not run on the background when no
  drone is detected. All five toggles are enabled initially; disable individual
  features to reduce inference work. Turning off Drone pauses feature models too.
  Models load on demand and are reused across sessions.
- Camera capture runs separately from detection. The live preview draws recent
  drone boxes as soon as the detector returns, before feature-model loading or
  inference finishes. The latest confirmed boxes remain visible until a new
  detection result replaces them; an empty result clears them. Results older
  than 0.5 seconds are amber and show their source-frame age rather than disappearing.
  Boxes are detection snapshots, not motion tracking. Resolution or model changes
  prevent incompatible overlays from being drawn. Feature labels cannot cover
  box outlines, which are drawn last. Each box includes
  every feature and its available confidence score. The feature table shows
  drone confidence, angle, distance, LED ID, and speed for each detected drone.
  Unavailable, pending, or disabled feature estimates show **N/A** in both the
  preview labels and the table. When no drone is detected, all table values
  become **N/A**. Additional drone rows can be scrolled without hiding columns.
- **Live FPS target** in the Camera tab sets the live capture/preview target
  from 1 to 240 FPS, including fractional rates such as 29.97. Change it while
  the camera is running: arrows apply immediately, and typed values apply after
  a short pause or on Enter. The camera session stays open. Invalid input keeps
  the last valid target. Actual FPS depends on the camera, exposure, and processing
  capacity; the preview reports the measured rate alongside the target.
- **Recording playback FPS** is a separate setting in the Recording tab, fixed
  for the current session. Live rate changes leave an active recording's FPS intact.
  **Save analysis video** saves full frames with drone boxes and features at detection speed.
  If detection runs below the chosen FPS, that recording plays faster than real
  time. The preview shows separate capture and detection rates.
- The camera starts with automatic exposure, avoiding an unintended dark image
  from a fixed short shutter time. **Manual camera exposure** sends your chosen
  settings to the camera when enabled. **Shutter (1/s)** is
  reciprocal exposure time: `1000` requests a 1/1000-second exposure. It is not
  the sensor's row-scan frequency. Turning the checkbox off requests automatic
  exposure and ISO. Changes apply after typing pauses or on **Apply exposure**.
- Linux exposure values are clamped to the camera's supported range and step;
  the reported exposure is shown below the controls. ISO is applied only when
  the driver exposes a standard ISO control, using its nearest supported value.
  Unsupported controls are reported explicitly. No artificial stripes, fake ISO
  gain or automatic software brightness correction are added to captured images.
  Real rolling-shutter bands depend on the sensor, exposure time and LED modulation.
  See the [Linux camera control reference](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html)
  for the exposure and ISO units used by the driver.
- GNU/Linux live feature inference uses each loaded checkpoint's training exposure. The
  bundled **Outdoor Angle, Distance, LED ID and Speed** checkpoints all use
  **1/6000 s, ISO 1000**, as recorded for their March 3, 2026 training run.
  Earlier ISO 800 settings (including the older 1/6400 s speed dataset) belong
  to earlier datasets and are not applied to these Outdoor checkpoints.
  The Detection tab lists the presets and the camera's actual feature exposure.
  The Indoor/Outdoor **drone** selector continues to select only the drone model.
- After finding a drone in a regular-exposure frame, the capture worker requests
  a separate real frame at the feature model's exposure, discards transition
  frames, then restores the user's preview/drone exposure. Models with the same
  preset share that capture; different presets use separate captures. Feature
  images are cropped to the drone's last detected location and are captured
  slightly later, so rapid motion can reduce feature accuracy. Preview holds its
  last normal frame during the exposure switch, then continues while inference
  runs. No synthetic rolling-shutter stripes or artificial ISO gain are applied.
  The camera may round exposure or lack ISO control; the actual result is shown,
  and exact agreement with the training settings depends on the camera hardware.
- **Logging is off by default.** In **Logs**, choose a CSV destination and click
  **Start logging** to opt in. Merely opening the app, choosing a file, or starting
  the camera does not create a log. **Stop logging** finishes the file while the
  camera keeps running. Video recording is independent of logging.
  Logs include UTC capture and logging timestamps, session/frame numbers, drone
  model, bounding boxes, confidence values (0–1), and all four feature estimates.
  Each detected drone gets a row; frames without a drone get a row with **N/A**
  values. Missing or disabled feature estimates also use **N/A**.
  The capture timestamp identifies the drone-detection frame; live feature
  estimates come from the subsequent exposure-specific capture described above.
  A compatible CSV file is appended to, preserving existing rows; unrelated files
  are rejected. File writes run separately from capture and inference. Storage
  errors or an excessive backlog stop logging and appear in Logs, without stopping
  the camera. The old automatic stdout-to-text-file logging is disabled in the GUI.
- Recordings and optional logs default to
  `${XDG_DATA_HOME:-$HOME/.local/share}/posefly/results/`. **Choose output file**
  selects the video destination; **Choose log file** selects the CSV destination.
  Reusing a video filename replaces that recording; log rows are preserved.
- **Stop camera** finishes recording and any active log. Logging remains off
  after restarting capture until **Start logging** is clicked again.
  Closing the main window waits for capture, detection, and pending log writes
  to finish and releases the output files before exiting. The
  trajectory viewer is not launched automatically.

If no camera is found, check that `/dev/video*` exists, close other camera apps,
and check your account's device access (`ls -l /dev/video*`, `id`). Device access
is normally granted by the desktop session or the distribution's `video` group.
Do not run the application as root. A driver that blocks indefinitely during a
camera read can still delay worker shutdown.

Drone inference now uses the entire camera image at YOLO's 640-pixel inference
size, preserving aspect ratio with padding. No center crop discards side regions;
returned boxes already use the original camera coordinates.

## Repository cleanup

Test and experiment folders are omitted from this checkout. Generated Python and
tool caches are ignored; the GNU/Linux launcher disables Python bytecode writes.
Training datasets and model weights are retained.

### Drone checkpoint validation

On the repository's 40 labelled test images, at confidence 0.10 and an IoU
matching threshold of 0.50, the previous custom-attention/512-center-crop path
matched 1 of 40 annotated drones. The Indoor checkpoint using the full frame at
640 matched 40 of 40, with mean matched IoU 0.935. The standard Outdoor checkpoint
at 640 matched 28 of 40. These are results on this included indoor dataset, not
an estimate of accuracy in arbitrary scenes or of false positives on background
footage. Original checkpoint files are preserved.
