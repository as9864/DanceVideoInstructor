"""Report-layer video writer (ARCHITECTURE.md §9: OpenCV for rendering).

Overlay rendering doesn't need PyAV's timestamp precision — it just needs to
write frames out in order at a fixed fps.
"""

from pathlib import Path

import cv2
import numpy as np


class VideoWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int, fourcc: str = "mp4v"):
        self.path = Path(path)
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"failed to open VideoWriter for {self.path}")

    def write(self, image_rgb: np.ndarray) -> None:
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        self._writer.write(bgr)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
