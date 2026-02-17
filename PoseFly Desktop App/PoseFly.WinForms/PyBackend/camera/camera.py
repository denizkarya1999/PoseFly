# camera.py
import os
import cv2
import math
import numpy as np

# ============================================================
# OPTION A (Outdoor-stable video):
# - Do NOT force dark
# - Keep hardware AUTO exposure enabled
# - Use only SOFTWARE brightness polish (HSV V multiplier)
# - Avoid fighting the driver with manual exposure nudging
# - Rolling shutter logic is kept EXACTLY as-is (unchanged),
#   but you can disable calling it via ENABLE_ROLLING_SHUTTER.
# ============================================================

# --- Brightness (hardware baseline) ---
BASE_BRIGHTNESS = 0.0

# --- Auto exposure/brightness tuning (software AE polish) ---
AUTO_TARGET_V = 110.0        # target brightness level in HSV V (0..255)
AUTO_ALPHA = 0.08            # smoothing (0..1). lower = more stable
AUTO_MIN_MUL = 0.35
AUTO_MAX_MUL = 1.60          # keep smaller max to avoid blowout

AUTO_ROI_TOP = 0.18          # ignore top overlay (18%)
AUTO_ROI_BORDER = 0.12       # ignore left/right/bottom borders (12%)

AUTO_SAT_V = 250             # saturation threshold in HSV V
AUTO_MIN_VALID_PIXELS = 2000

# Rate-limit multiplier to avoid oscillations (per update)
AUTO_MAX_STEP_UP = 1.03      # +3% per update
AUTO_MAX_STEP_DN = 1.0 / 1.03

# Hardware exposure nudging (best-effort; driver-dependent)
# OPTION A: disable (don’t fight the driver outdoors)
HW_AE_ENABLE = False
HW_AE_EVERY_N_FRAMES = 15
HW_ERR_DEADZONE = 8.0
HW_EXPOSURE_STEP = 0.10      # small step to avoid oscillation
HW_EXPOSURE_MIN = -13.0
HW_EXPOSURE_MAX = -1.0

# Optional: lock auto after warmup (helps ML stability)
# OPTION A: keep software AE running continuously
AE_WARMUP_FRAMES = 0         # set 0 to disable locking

# --- HARD DARK MODE (force as dark as possible) ---
# OPTION A: OFF
FORCE_DARK = False

# software darkening (guaranteed)
DARK_V_MUL = 0.10
DARK_GAMMA = 3.5

# hardware forcing (best-effort; driver dependent)
HW_FORCE_EXPOSURE = -13.0
HW_FORCE_GAIN = 0.0
HW_FORCE_BRIGHTNESS = 0.0
HW_FORCE_GAMMA = 3.0

# ============================================================
# Rolling shutter toggle (OPTION A):
# Keep the rolling shutter FUNCTION untouched, but
# prevent your backend from applying it in outdoor mode.
# ============================================================
ENABLE_ROLLING_SHUTTER = False

class Camera:
    def __init__(self):
        self.cap = None
        self.w = None
        self.h = None

        # ---- Rolling shutter / ISO state ----
        self.iso = 250
        self.shutter_hz = 1000.0

        # Precomputed row gain for OOK stripes (built in _apply_rollingshutter)
        self._ook_row_gain = None

        # ---- Writer state ----
        self.out = None
        self.fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.last_writer_path = None
        self.last_writer_fps = None

        # ---- Auto brightness/exposure state ----
        self.brightness_mul = 1.0
        self.auto_ae = True
        self._frame_i = 0

        # Track whether we set manual exposure mode once (avoid flipping modes)
        self._hw_manual_set = False

        # Precomputed gamma LUT for FORCE_DARK
        self._dark_lut = None

    # ---------------- Public controls ----------------
    def set_brightness(self, mul: float):
        """Manual override for software brightness multiplier."""
        self.brightness_mul = float(max(0.1, min(3.0, mul)))

    def set_auto_ae(self, enabled: bool):
        """Enable/disable software auto brightness + optional hardware nudging."""
        self.auto_ae = bool(enabled)

    # ---------------- Auto control helpers ----------------
    def _auto_adjust_brightness(self, frame_bgr):
        """
        Robust software AE:
        - uses ROI (ignores overlay + borders)
        - ignores saturated pixels
        - percentile-based measurement for robustness
        - EMA + rate-limit to reduce oscillation
        Returns: (err, sat_ratio, used_pixels)
        """
        h, w = frame_bgr.shape[:2]

        y0 = int(h * AUTO_ROI_TOP)
        y1 = int(h * (1.0 - AUTO_ROI_BORDER))
        x0 = int(w * AUTO_ROI_BORDER)
        x1 = int(w * (1.0 - AUTO_ROI_BORDER))

        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return 0.0, 0.0, 0

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)

        sat = (v >= AUTO_SAT_V)
        sat_ratio = float(sat.mean())

        valid = v[~sat]
        if valid.size < AUTO_MIN_VALID_PIXELS:
            mean_v = float(np.mean(v))
            err = AUTO_TARGET_V - mean_v
            return err, sat_ratio, int(valid.size)

        # Robust brightness proxy
        level = float(np.percentile(valid, 60))
        err = AUTO_TARGET_V - level

        desired_mul = AUTO_TARGET_V / max(1.0, level)

        # Rate limit to prevent sudden swings
        desired_mul = max(self.brightness_mul * AUTO_MAX_STEP_DN,
                          min(self.brightness_mul * AUTO_MAX_STEP_UP, desired_mul))

        # Smooth update (EMA)
        self.brightness_mul = (1.0 - AUTO_ALPHA) * self.brightness_mul + AUTO_ALPHA * desired_mul
        self.brightness_mul = float(max(AUTO_MIN_MUL, min(AUTO_MAX_MUL, self.brightness_mul)))

        return err, sat_ratio, int(valid.size)

    def _ensure_hw_manual_once(self):
        """Set hardware exposure mode to MANUAL once (DShow typical), without flipping repeatedly."""
        if self.cap is None or self._hw_manual_set:
            return
        try:
            # DShow: 0.25 tends to mean MANUAL, 0.75 tends to mean AUTO
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self._hw_manual_set = True
        except Exception:
            pass

    def _maybe_nudge_hw_exposure(self, err, sat_ratio):
        """
        Best-effort hardware exposure adjustment.
        Assumes MANUAL mode already set once; does not toggle AUTO/MANUAL in the loop.
        """
        if self.cap is None:
            return

        # Only act if clearly off
        if abs(err) < HW_ERR_DEADZONE and sat_ratio < 0.25:
            return

        try:
            cur = float(self.cap.get(cv2.CAP_PROP_EXPOSURE))
            if not np.isfinite(cur):
                return

            # If saturated / too bright -> decrease exposure (more negative)
            if sat_ratio > 0.35 or err < 0:
                cur -= HW_EXPOSURE_STEP
            else:
                cur += HW_EXPOSURE_STEP

            cur = max(HW_EXPOSURE_MIN, min(HW_EXPOSURE_MAX, cur))
            self.cap.set(cv2.CAP_PROP_EXPOSURE, cur)

            # Optional: gently reduce gain when saturated (driver-dependent)
            if sat_ratio > 0.35:
                g = float(self.cap.get(cv2.CAP_PROP_GAIN))
                if np.isfinite(g):
                    self.cap.set(cv2.CAP_PROP_GAIN, max(0.0, g - 1.0))
        except Exception:
            pass

    def _build_dark_lut(self):
        """Build LUT for gamma darkening when FORCE_DARK is enabled."""
        inv_gamma = 1.0 / max(0.1, float(DARK_GAMMA))
        lut = (np.linspace(0, 1, 256) ** inv_gamma) * 255.0
        self._dark_lut = lut.astype(np.uint8)

    def _force_dark_hw_once(self):
        """Force hardware settings to be as dark as possible (best-effort)."""
        if self.cap is None:
            return

        # Set manual exposure mode once
        self._ensure_hw_manual_once()

        # Push the camera to minimum light (if supported)
        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(HW_FORCE_EXPOSURE))
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_GAIN, float(HW_FORCE_GAIN))
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, float(HW_FORCE_BRIGHTNESS))
        except Exception:
            pass
        try:
            self.cap.set(cv2.CAP_PROP_GAMMA, float(HW_FORCE_GAMMA))
        except Exception:
            pass

    # ---------------- Camera lifecycle ----------------
    def open(self, camera_index=0, use_dshow=True):
        if self.cap is not None:
            return

        self.cap = (
            cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if use_dshow else
            cv2.VideoCapture(camera_index)
        )

        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError("Could not open webcam.")

        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # OPTION A: keep hardware AUTO exposure enabled (don’t fight the driver)
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except Exception:
            pass

        # Reset counters
        self._frame_i = 0
        self._hw_manual_set = False

        # Build LUT (unused in OPTION A because FORCE_DARK=False)
        self._build_dark_lut()

        # If FORCE_DARK, immediately push hardware as dark as possible and
        # disable the adaptive controllers so nothing brightens later.
        if FORCE_DARK:
            self._force_dark_hw_once()
            self.auto_ae = False
            self.brightness_mul = 1.0
        else:
            # OPTION A: keep software AE running
            self.auto_ae = True
            self.brightness_mul = 1.0

    def apply_led_settings(self):
        """
        Backwards-compatible function.
        Your backend/server calls this during START.
        OPTION A: no-op unless ENABLE_ROLLING_SHUTTER=True.
        """
        if self.cap is None:
            raise RuntimeError("Camera not opened.")
        if not ENABLE_ROLLING_SHUTTER:
            return
        self._apply_rollingshutter()

    def rollingshutter(self, iso: int, shutter_hz: float):
        """
        Set ISO-like + shutter rate (Hz) and apply.
        OPTION A: store values but only apply if ENABLE_ROLLING_SHUTTER=True.
        """
        self.iso = int(max(50, min(6400, iso)))
        self.shutter_hz = float(max(5.0, min(6000.0, shutter_hz)))
        if not ENABLE_ROLLING_SHUTTER:
            return
        self._apply_rollingshutter()

    # ---------------- Rolling shutter (KEEP) ----------------
    def _apply_rollingshutter(self):
        """
        Keeps your original rolling shutter logic.
        NOTE: This sets CAP_PROP_GAIN / EXPOSURE / BRIGHTNESS / GAMMA.
        If you want hardware auto exposure, avoid calling this repeatedly.
        """
        if self.cap is None:
            return

        # Keep original behavior
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        # --- Camera-like mapping (matches apply_rolling_shutter) ---

        # ISO -> gain (log mapping)
        iso_f = float(max(1, self.iso))
        t_iso = (math.log(iso_f) - math.log(50.0)) / (math.log(6400.0) - math.log(50.0))
        t_iso = max(0.0, min(1.0, t_iso))
        gain = 2.0 + t_iso * 18.0  # ~2..20

        # shutter_hz -> exposure proxy (higher Hz => darker)
        sh_f = float(max(1, self.shutter_hz))
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

        row_time = 1.0 / max(1, int(self.shutter_hz))

        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if h > 0:
            y = np.arange(h, dtype=np.float32)
            t = y * row_time
            phase = (t * float(led_freq_hz)) % 1.0
            on = (phase < float(duty)).astype(np.float32)

            row_gain = (1.0 - float(contrast)) + float(contrast) * on
            self._ook_row_gain = row_gain[:, None]
        else:
            self._ook_row_gain = None

        # Apply what the camera can accept (global controls)
        self.cap.set(cv2.CAP_PROP_GAIN, float(gain))
        self.cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure))

        brightness_prop = BASE_BRIGHTNESS + 0.5 * (brightness_scale - 1.0)
        brightness_prop = max(0.0, min(1.0, brightness_prop))
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, float(brightness_prop))

        gamma = 1.0 / max(0.1, exposure_scale * brightness_scale)
        gamma = max(0.3, min(3.0, gamma))
        self.cap.set(cv2.CAP_PROP_GAMMA, float(gamma))

    # ---------------- Frame read ----------------
    def read(self):
        if self.cap is None:
            return False, None

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return ok, frame

        self._frame_i += 1

        # 1) Apply stripe effect (rolling shutter simulation) if enabled
        if self._ook_row_gain is not None:
            if self._ook_row_gain.shape[0] != frame.shape[0]:
                self._apply_rollingshutter()

            if self._ook_row_gain is not None and self._ook_row_gain.shape[0] == frame.shape[0]:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 2] *= self._ook_row_gain
                hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
                frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 2) Auto control (software AE + optional HW nudging)
        if self.auto_ae and not FORCE_DARK:
            err, sat_ratio, _ = self._auto_adjust_brightness(frame)

            # Optional warmup lock (disabled in OPTION A by AE_WARMUP_FRAMES=0)
            if AE_WARMUP_FRAMES > 0 and self._frame_i >= AE_WARMUP_FRAMES:
                self.auto_ae = False
            else:
                if HW_AE_ENABLE and (self._frame_i % HW_AE_EVERY_N_FRAMES == 0):
                    self._ensure_hw_manual_once()
                    self._maybe_nudge_hw_exposure(err, sat_ratio)

        # 3) Apply software brightness last (polish)
        if (self.brightness_mul != 1.0) and (not FORCE_DARK):
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] *= self.brightness_mul
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # 4) HARD DARK MODE (disabled in OPTION A by FORCE_DARK=False)
        if FORCE_DARK:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 2] *= float(DARK_V_MUL)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
            frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            if self._dark_lut is None:
                self._build_dark_lut()
            frame = cv2.LUT(frame, self._dark_lut)

        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # ---------------- Writer ----------------
    def _ensure_writer(self, path, fps):
        if not path:
            return

        if (
            self.out is not None and
            self.last_writer_path == path and
            self.last_writer_fps == fps
        ):
            return

        self.release_writer()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.out = cv2.VideoWriter(path, self.fourcc, fps, (self.w, self.h))
        if not self.out.isOpened():
            self.out = None
            raise RuntimeError("Failed to open VideoWriter.")

        self.last_writer_path = path
        self.last_writer_fps = fps

    def write_if_enabled(self, frame, enabled, path, fps):
        if not enabled:
            self.release_writer()
            return
        self._ensure_writer(path, fps)
        self.out.write(frame)

    def release_writer(self):
        if self.out is not None:
            self.out.release()
            self.out = None
        self.last_writer_path = None
        self.last_writer_fps = None