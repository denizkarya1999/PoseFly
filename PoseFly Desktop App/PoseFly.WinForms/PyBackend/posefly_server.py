import base64
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback

import cv2

from backend import PipelineBackend

HOST = "127.0.0.1"
PORT = 8765


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

    ts = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join("results", f"posefly_server_{ts}_results_only.txt")


class PoseFlyServer:
    def __init__(self):
        self.backend = PipelineBackend()
        self.client_sock = None
        self.client_lock = threading.Lock()

        self.running = False
        self.thread = None

        # Runtime settings
        self.camera_index = 0
        self.use_dshow = True
        self.fps = 60.0
        self.output_path = "results/full_pipeline_results/posefly_results.mp4"
        self.save_video = True
        self.toggles = {
            "drone": True,
            "angle": True,
            "distance": True,
            "led": True,
            "speed": True,
        }

        # Rolling shutter controls
        self.iso = 250
        self.shutter_hz = 1000.0

        # Results logging / trajectory
        self.results_log_path = make_results_log_path(self.output_path)
        self.trajectory_launched = False

    def _send_line(self, line: str):
        with self.client_lock:
            if not self.client_sock:
                return
            try:
                self.client_sock.sendall((line + "\n").encode("utf-8"))
            except Exception:
                self.client_sock = None

    def _launch_trajectory_algorithm(self, log_file_path: str):
        if self.trajectory_launched:
            return

        if not log_file_path or not os.path.exists(log_file_path):
            self._send_line("STATUS Trajectory skipped: log file not found")
            return

        if os.path.getsize(log_file_path) <= 0:
            self._send_line("STATUS Trajectory skipped: log file is empty")
            return

        here = os.path.dirname(os.path.abspath(__file__))
        trajectory_script = os.path.join(here, "videoprocessing", "tools", "trajectory_algorithm.py")

        if not os.path.isfile(trajectory_script):
            self._send_line(f"STATUS Trajectory script not found: {trajectory_script}")
            return

        cmd = [sys.executable, trajectory_script, log_file_path]

        try:
            subprocess.Popen(cmd)
            self.trajectory_launched = True
            self._send_line("STATUS Trajectory launched")
        except Exception as e:
            self._send_line(f"STATUS Failed to launch trajectory: {e}")

    def apply_update(self, payload: dict):
        if "camera_index" in payload:
            self.camera_index = int(payload["camera_index"])
        if "use_dshow" in payload:
            self.use_dshow = bool(payload["use_dshow"])
        if "fps" in payload:
            self.fps = float(payload["fps"])
        if "output_path" in payload:
            self.output_path = str(payload["output_path"])
        if "save_video" in payload:
            self.save_video = bool(payload["save_video"])

        if "toggles" in payload:
            t = payload["toggles"] or {}
            self.toggles = {
                "drone": bool(t.get("drone", True)),
                "angle": bool(t.get("angle", True)),
                "distance": bool(t.get("distance", True)),
                "led": bool(t.get("led", True)),
                "speed": bool(t.get("speed", True)),
            }

        if "iso" in payload:
            self.iso = int(payload["iso"])
        if "shutter_hz" in payload:
            self.shutter_hz = float(payload["shutter_hz"])

        # Refresh log path whenever output path changes
        self.results_log_path = make_results_log_path(self.output_path)

        # Apply immediately if camera is open
        try:
            self.backend.set_rollingshutter(self.iso, self.shutter_hz)
        except Exception:
            pass

        self._send_line("STATUS UPDATED")

    def _capture_loop(self):
        old_stdout = sys.stdout
        results_logger = None

        try:
            os.makedirs(os.path.dirname(self.results_log_path) or ".", exist_ok=True)
            results_logger = ResultsOnlyLogger(self.results_log_path, keep_console=True)
            sys.stdout = results_logger
        except Exception as e:
            sys.stdout = old_stdout
            self._send_line(f"STATUS WARN could not start results logger: {e}")
            results_logger = None

        try:
            self.backend.open_camera(camera_index=self.camera_index, use_dshow=self.use_dshow)
            self.backend.apply_camera_settings_led_id()
            self.backend.set_rollingshutter(self.iso, self.shutter_hz)
        except Exception as e:
            self._send_line(f"STATUS ERROR {e}")
            self.running = False

            try:
                sys.stdout = old_stdout
            except Exception:
                pass
            try:
                if results_logger is not None:
                    results_logger.close()
            except Exception:
                pass
            return

        self._send_line(f"STATUS RUNNING LOG={self.results_log_path}")

        target_dt = 1.0 / max(self.fps, 0.1)

        try:
            while self.running:
                t0 = time.time()
                try:
                    ret, frame = self.backend.read_frame()
                    if not ret or frame is None:
                        self._send_line("STATUS ERROR Failed to read frame")
                        break

                    out = self.backend.process_frame(frame, self.toggles)

                    if isinstance(out, tuple) and len(out) == 2:
                        frame = out[1]
                    elif out is not None:
                        frame = out

                    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ok:
                        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                        self._send_line("FRAME " + b64)

                    try:
                        self.backend.write_frame_if_enabled(frame, self.save_video, self.output_path, self.fps)
                    except Exception as e:
                        self.save_video = False
                        self._send_line(f"STATUS WARN Video writer disabled: {e}")
                        try:
                            self.backend.release_writer()
                        except Exception:
                            pass

                except Exception as e:
                    print(traceback.format_exc())
                    self._send_line(f"STATUS ERROR {e}")
                    break

                elapsed = time.time() - t0
                sleep_for = max(0.0, target_dt - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)

        finally:
            self.running = False

            try:
                self.backend.release_writer()
            except Exception:
                pass
            try:
                self.backend.release_camera()
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

        self._send_line("STATUS STOPPED")

        try:
            self._launch_trajectory_algorithm(self.results_log_path)
        except Exception as e:
            self._send_line(f"STATUS Trajectory launch error: {e}")

    def start_pipeline(self):
        if self.running:
            return
        self.trajectory_launched = False
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop_pipeline(self):
        self.running = False

    def handle_client(self, sock: socket.socket):
        with self.client_lock:
            self.client_sock = sock

        self._send_line("STATUS CONNECTED")

        f = sock.makefile("r", encoding="utf-8", newline="\n")
        try:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except Exception:
                    self._send_line("STATUS ERROR Bad JSON")
                    continue

                cmd = (msg.get("cmd") or "").upper()
                payload = msg.get("payload") or {}

                if cmd == "START":
                    self.apply_update(payload)
                    self.start_pipeline()
                elif cmd == "UPDATE":
                    self.apply_update(payload)
                elif cmd == "STOP":
                    self.stop_pipeline()
                elif cmd == "PING":
                    self._send_line("STATUS PONG")
                else:
                    self._send_line(f"STATUS ERROR Unknown cmd: {cmd}")

        except (ConnectionResetError, BrokenPipeError, OSError):
            pass
        finally:
            self.stop_pipeline()
            with self.client_lock:
                try:
                    sock.close()
                except Exception:
                    pass
                self.client_sock = None


def main():
    server = PoseFlyServer()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"PoseFly Python server listening on {HOST}:{PORT}")

        while True:
            client, addr = s.accept()
            print("Client connected:", addr)
            try:
                server.handle_client(client)
            except Exception as e:
                print("Client handler error:", repr(e))
            print("Client disconnected:", addr)


if __name__ == "__main__":
    main()