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
works after moving the whole project directory. Use **PoseFly** to launch from
the folder; there is no separate `PoseFly.desktop` file in the project.

Launch from the project folder; PoseFly stays out of the GNOME application menu.
The launcher registers a **hidden** `posefly.desktop` (`NoDisplay=true`) and the
bundled icon under `${XDG_DATA_HOME:-$HOME/.local/share}` solely for taskbar icon
matching. The window class and hidden desktop entry both identify **PoseFly**.
Registration is refreshed after a move; `python3 linux/install_desktop.py` also
updates an existing installation to hide its menu entry.
No administrator access is needed. This follows the
[desktop entry specification](https://specifications.freedesktop.org/desktop-entry/latest-single/)
for `Icon`, `StartupWMClass` and `NoDisplay`.

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
  on that drone's crop with the internal preprocessing described below. They do
  not run on the background when no drone is detected. All five toggles are
  enabled initially; disable individual
  features to reduce inference work. Turning off Drone pauses feature models too.
  Models load on demand and are reused across sessions.
- In **Detection → Model settings**, set a separate **Confidence (%)** for
  Drone, Angle, Distance, LED ID and Speed. Enter 0–100%; arrows, Enter, leaving
  the field, or **Apply settings** apply the settings live. Typed changes also
  apply after a short pause. Higher thresholds reject more predictions; feature
  estimates below their threshold show **N/A**. Results from older settings are
  hidden while the next inference completes. Invalid input keeps the last valid
  thresholds. **Reset defaults** restores Drone to 25% and each feature model
  to 0.1%. Settings last for the current app
  session; changing them does not restart the camera or change exposure.
- **Max boxes** beside each confidence field sets a whole-number maximum from
  1 to 300. The Drone limit applies to the full frame and caps the drone boxes
  shown in the preview, result table, recordings and optional logs. Each feature
  limit applies per detected drone crop; all returned candidates are considered
  and the strongest estimate is shown for that feature. Counts are maximums,
  not a request to invent detections: confidence filtering may return fewer.
  Changes apply live, with old results hidden until updated inference returns.
  **Reset defaults** restores one drone box and one box per feature model,
  together with the default confidence thresholds. Counts last for this session.
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
  from a fixed short shutter time. **Manual drone exposure** sends your chosen
  settings to the camera when enabled. **Shutter (1/s)** is
  reciprocal exposure time: `1000` requests a 1/1000-second exposure. It is not
  the sensor's row-scan frequency. Turning the checkbox off requests automatic
  exposure and ISO. Changes apply after typing pauses or on **Apply exposure**.
- Linux exposure values are clamped to the camera's supported range and step;
  the reported exposure is shown below the controls. ISO is applied only when
  the driver exposes a standard ISO control, using its nearest supported value.
  Unsupported controls are reported explicitly. The live preview and drone
  detector receive the captured image without artificial exposure effects.
  Real rolling-shutter bands depend on the sensor, exposure time and LED modulation.
  See the [Linux camera control reference](https://docs.kernel.org/userspace-api/media/v4l/ext-ctrls-camera.html)
  for the exposure and ISO units used by the driver.
- The camera uses the exposure selected in **Drone camera exposure** throughout
  live processing. PoseFly applies that choice at startup and when you change it
  or explicitly select **Apply exposure**. Feature models never switch the
  camera's shutter rate or ISO. With automatic exposure selected, the camera
  manages exposure. The preview continues independently of inference without
  exposure-switch pauses.
  Angle, Distance, LED ID and Speed each receive their own drone crop from that
  frame, with an artificial shutter/ISO effect applied immediately before model
  inference. Each model uses its own internal training profile; the current
  outdoor checkpoints all use 1/6000 s and ISO 1000. These settings and processed
  images stay in the background. Only drone camera exposure controls are visible.
  The software effect approximates brightness and bands using a fixed 2 kHz,
  50% duty reference pattern. It cannot recreate a physical exposure or recover
  motion/LED information absent from the capture; feature accuracy still depends
  on the captured image. It never alters the source frame or drone detector input.
  The Indoor/Outdoor **drone** selector continues to select only the drone model.
- **Logging is off by default.** In **Logs**, choose a CSV destination and click
  **Start logging** to opt in. Merely opening the app, choosing a file, or starting
  the camera does not create a log. **Stop logging** finishes the file while the
  camera keeps running. Video recording is independent of logging.
  Logs include UTC capture and logging timestamps, session/frame numbers, drone
  model, bounding boxes, confidence values (0–1), and all four feature estimates.
  Each detected drone gets a row; frames without a drone get a row with **N/A**
  values. Missing or disabled feature estimates also use **N/A**.
  The capture timestamp identifies the shared source frame for drone and feature
  estimates.
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
