"""Where the live HUD goes: an OpenCV window, a browser page, or nowhere.

The project depends on opencv-python-headless (no GUI — ARCHITECTURE.md
§9: OpenCV is only used for rendering), so `cv2.imshow` isn't available by
default. The browser display needs nothing extra: a stdlib HTTP server
streams the HUD as MJPEG, which any browser shows as a live image. With
`--host 0.0.0.0` a phone or tablet propped up in the studio can show the
feedback while the laptop with the GPU sits out of the way.
"""

import http.server
import queue
import socketserver
import threading
import webbrowser
from typing import Literal, Protocol

import cv2
import numpy as np

Command = Literal["quit", "recalibrate"]


class Display(Protocol):
    def show(self, image_rgb: np.ndarray) -> None: ...
    def poll_command(self) -> Command | None: ...
    def close(self) -> None: ...


def gui_available() -> bool:
    info = cv2.getBuildInformation()
    for line in info.splitlines():
        if line.strip().startswith("GUI:"):
            return "NONE" not in line.upper()
    return False


class NullDisplay:
    def show(self, image_rgb: np.ndarray) -> None:
        pass

    def poll_command(self) -> Command | None:
        return None

    def close(self) -> None:
        pass


class WindowDisplay:
    """cv2.imshow window. Keys: q/Esc quit, c recalibrate."""

    def __init__(self, title: str = "dvi live"):
        self.title = title
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        self._pending: Command | None = None

    def show(self, image_rgb: np.ndarray) -> None:
        cv2.imshow(self.title, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            self._pending = "quit"
        elif key == ord("c"):
            self._pending = "recalibrate"
        elif cv2.getWindowProperty(self.title, cv2.WND_PROP_VISIBLE) < 1:
            self._pending = "quit"

    def poll_command(self) -> Command | None:
        cmd, self._pending = self._pending, None
        return cmd

    def close(self) -> None:
        cv2.destroyWindow(self.title)


_PAGE = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>dvi live</title>
<style>
 :root { color-scheme: dark; }
 body { margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif;
        display: flex; flex-direction: column; align-items: center; min-height: 100vh; }
 img { max-width: 100%; max-height: calc(100vh - 64px); object-fit: contain; }
 .bar { display: flex; gap: 12px; padding: 12px; }
 button { font-size: 16px; padding: 8px 16px; border-radius: 8px; border: 1px solid #555;
          background: #222; color: #eee; cursor: pointer; }
 button:hover { background: #333; }
</style></head>
<body>
<img src="/stream.mjpg" alt="dvi live HUD">
<div class="bar">
 <button onclick="fetch('/recalibrate',{method:'POST'})">기준 자세 다시 측정 (c)</button>
 <button onclick="fetch('/quit',{method:'POST'}).then(()=>document.body.innerHTML='<p>종료했습니다.</p>')">종료 (q)</button>
</div>
<script>
document.addEventListener('keydown', e => {
  if (e.key === 'c') fetch('/recalibrate', {method: 'POST'});
  if (e.key === 'q') fetch('/quit', {method: 'POST'});
});
</script>
</body></html>
"""


class _ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class BrowserDisplay:
    """MJPEG-over-HTTP HUD with recalibrate/quit buttons."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True, jpeg_quality: int = 80):
        self.jpeg_quality = jpeg_quality
        self._frame_cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0
        self._closed = False
        self._commands: queue.Queue[Command] = queue.Queue()
        display = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep the console for coaching cues
                pass

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = _PAGE.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    display._stream_to(self.wfile)
                else:
                    self.send_error(404)

            def do_POST(self):
                if self.path in ("/recalibrate", "/quit"):
                    display._commands.put(self.path.lstrip("/"))
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_error(404)

        self._server = _ThreadingServer((host, port), Handler)
        self.url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{self._server.server_address[1]}/"
        self._thread = threading.Thread(target=self._server.serve_forever, name="dvi-http", daemon=True)
        self._thread.start()
        if open_browser:
            webbrowser.open(self.url)

    def _stream_to(self, wfile) -> None:
        last_seq = -1
        while True:
            with self._frame_cond:
                while self._seq == last_seq and not self._closed:
                    self._frame_cond.wait(timeout=1.0)
                if self._closed:
                    return
                jpeg, last_seq = self._jpeg, self._seq
            try:
                wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                wfile.write(jpeg)
                wfile.write(b"\r\n")
                wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                return  # browser tab closed

    def show(self, image_rgb: np.ndarray) -> None:
        ok, buf = cv2.imencode(
            ".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            return
        with self._frame_cond:
            self._jpeg = buf.tobytes()
            self._seq += 1
            self._frame_cond.notify_all()

    def poll_command(self) -> Command | None:
        try:
            return self._commands.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        with self._frame_cond:
            self._closed = True
            self._frame_cond.notify_all()
        self._server.shutdown()
        self._server.server_close()
