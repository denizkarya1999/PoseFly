# posefly_server.py
import base64
import json
import socket
import threading
import time
import cv2

from backend import PipelineBackend  # your backend.py

HOST = "127.0.0.1"
PORT = 8765


class PoseFlyServer:
    def __init__(self):
        self.backend = PipelineBackend()
        self.client_sock = None
        self.client_lock = threading.Lock()

        self.running = False
        self.thread = None

        # Runtime settings (can be updated via UPDATE)
        self.camera_index = 0
        self.use_dshow = True
        self.fps = 10.0
        self.output_path = "results/full_pipeline_results/posefly_results2.mp4"
        self.save_video = True
        self.toggles = {"drone": True, "angle": True, "distance": True, "led": True}

    def _send_line(self, line: str):
        with self.client_lock:
            if not self.client_sock:
                return
            try:
                self.client_sock.sendall((line + "\n").encode("utf-8"))
            except Exception:
                # client disconnected
                self.client_sock = None

    def _capture_loop(self):
        # open camera once per START
        try:
            self.backend.open_camera(camera_index=self.camera_index, use_dshow=self.use_dshow)
            self.backend.apply_camera_settings_led_id()
        except Exception as e:
            self._send_line(f"STATUS ERROR {e}")
            self.running = False
            return

        self._send_line("STATUS RUNNING")

        target_dt = 1.0 / max(self.fps, 0.1)

        while self.running:
            t0 = time.time()
            try:
                ret, frame = self.backend.read_frame()
                if not ret:
                    self._send_line("STATUS ERROR Failed to read frame")
                    break

                # CV processing
                frame = self.backend.process_frame(frame, self.toggles)

                # Stream frame to client (JPEG -> base64)
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
                    self._send_line("FRAME " + b64)

                # Optional writer
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
                self._send_line(f"STATUS ERROR {e}")
                break

            # FPS pacing
            elapsed = time.time() - t0
            sleep_for = max(0.0, target_dt - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

        # cleanup
        self.running = False
        try:
            self.backend.release_writer()
        except Exception:
            pass
        try:
            self.backend.release_camera()
        except Exception:
            pass

        self._send_line("STATUS STOPPED")

    def start_pipeline(self):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop_pipeline(self):
        self.running = False

    def apply_update(self, payload: dict):
        # update settings safely mid-run
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
            }

        self._send_line("STATUS UPDATED")

    def handle_client(self, sock: socket.socket):
        with self.client_lock:
            self.client_sock = sock

        self._send_line("STATUS CONNECTED")

        # read JSON lines
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

        # ✅ IMPORTANT CHANGE:
        # Don't crash the whole server when a client disconnects abruptly (WinError 10054 is common).
        except (ConnectionResetError, BrokenPipeError, OSError):
            # client disconnected abruptly; treat as normal disconnect
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
                # ✅ IMPORTANT CHANGE:
                # Never let a single client/session crash the whole server loop.
                print("Client handler error:", repr(e))
            print("Client disconnected:", addr)


if __name__ == "__main__":
    main()