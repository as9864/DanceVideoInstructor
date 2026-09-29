"""Live frame sources for the stream runner (ARCHITECTURE.md §3 stage 1, §11).

Two very different latency contracts:

- LatestFrameCapture (webcam index, phone IP-camera URL, RTSP): a
  background thread keeps only the newest frame. If pose inference is
  slower than the camera, stale frames are dropped rather than queued —
  a queue would make feedback lag further behind the dancer every second
  (NFR-02: input -> feedback under 150ms).
- FileFrameSource (a recorded video): every frame in order, timestamps
  from the container. Deterministic, for trying `dvi live` on a clip.
"""

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from dvi.io.video_reader import VideoReader

Frame = tuple[np.ndarray, float]  # (RGB image, t seconds)


def _resize_long_edge(img: np.ndarray, max_long_edge: int | None) -> np.ndarray:
    if max_long_edge is None:
        return img
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if long_edge <= max_long_edge:
        return img
    s = max_long_edge / long_edge
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


class LatestFrameCapture:
    def __init__(
        self,
        source: int | str,
        *,
        width: int | None = None,
        height: int | None = None,
        max_long_edge: int | None = None,
        mirror: bool = False,
        open_timeout_s: float = 10.0,
    ):
        self.source = source
        self.max_long_edge = max_long_edge
        self.mirror = mirror
        self._cap = cv2.VideoCapture(source)
        if width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best effort; ignored by some backends
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera source {source!r}")

        self._lock = threading.Condition()
        self._frame: Frame | None = None
        self._seq = 0
        self._last_read_seq = 0
        self._stopped = False
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, name="dvi-capture", daemon=True)
        self._thread.start()

        deadline = time.monotonic() + open_timeout_s
        with self._lock:
            while self._frame is None and not self._stopped and time.monotonic() < deadline:
                self._lock.wait(timeout=0.1)
        if self._frame is None:
            self.close()
            raise RuntimeError(f"camera source {source!r} opened but delivered no frames")

    def _loop(self) -> None:
        while not self._stopped:
            ok, bgr = self._cap.read()
            t = time.monotonic() - self._t0  # capture time — basis for latency measurement
            if not ok:
                with self._lock:
                    self._stopped = True
                    self._lock.notify_all()
                return
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if self.mirror:
                img = np.ascontiguousarray(img[:, ::-1])
            img = _resize_long_edge(img, self.max_long_edge)
            with self._lock:
                self._frame = (img, t)
                self._seq += 1
                self._lock.notify_all()

    def now(self) -> float:
        """Same clock as frame timestamps, for measuring capture->display latency."""
        return time.monotonic() - self._t0

    def read(self, timeout_s: float = 2.0) -> Frame | None:
        """Block until a frame newer than the last one returned; None when the stream ended."""
        deadline = time.monotonic() + timeout_s
        with self._lock:
            while self._seq == self._last_read_seq and not self._stopped:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._lock.wait(timeout=remaining)
            if self._seq == self._last_read_seq:
                return None
            self._last_read_seq = self._seq
            return self._frame

    def close(self) -> None:
        self._stopped = True
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)
        self._cap.release()


class FileFrameSource:
    def __init__(self, path: Path, *, max_long_edge: int | None = None, mirror: bool = False):
        self.mirror = mirror
        self._reader = VideoReader(Path(path), max_long_edge=max_long_edge)
        self._iter = iter(self._reader)
        self._last_t = 0.0

    @property
    def fps(self) -> float:
        return self._reader.fps

    def now(self) -> float:
        # Offline clip: "latency" is just processing time, measured by the runner.
        return self._last_t

    def read(self, timeout_s: float = 2.0) -> Frame | None:
        vf = next(self._iter, None)
        if vf is None:
            return None
        img = np.ascontiguousarray(vf.image[:, ::-1]) if self.mirror else vf.image
        self._last_t = vf.t
        return img, vf.t

    def close(self) -> None:
        self._reader.close()


def parse_source(spec: str) -> int | str | Path:
    """'0' -> camera index 0; 'http://..' / 'rtsp://..' -> stream URL; else a file path."""
    if spec.isdigit():
        return int(spec)
    if "://" in spec:
        return spec
    return Path(spec)


def open_source(spec: str, *, max_long_edge: int | None = None, mirror: bool | None = None,
                width: int | None = None, height: int | None = None):
    """Mirror defaults to on for a local webcam (self-view, like a mirror in
    the studio) and off for files and network streams."""
    source = parse_source(spec)
    if isinstance(source, Path):
        if not source.exists():
            raise FileNotFoundError(source)
        return FileFrameSource(source, max_long_edge=max_long_edge, mirror=bool(mirror))
    if mirror is None:
        mirror = isinstance(source, int)
    return LatestFrameCapture(source, width=width, height=height, max_long_edge=max_long_edge, mirror=mirror)
