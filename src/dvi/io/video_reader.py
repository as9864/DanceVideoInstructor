"""Ingest stage (ARCHITECTURE.md §3 stage 1): decode, fix rotation, resize, normalize fps.

PyAV over OpenCV for reading — accurate per-frame timestamps matter because
every downstream time series is indexed by them (ARCHITECTURE.md §9).
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np


@dataclass
class VideoFrame:
    index: int
    t: float  # seconds
    image: np.ndarray  # (H, W, 3) uint8, RGB


class VideoReader:
    def __init__(
        self,
        path: Path,
        max_long_edge: int | None = None,
        target_fps: float | None = None,
    ):
        self.path = Path(path)
        self.max_long_edge = max_long_edge
        self.target_fps = target_fps

        self._container = av.open(str(self.path))
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"
        self._rotation = self._detect_rotation()

    def _detect_rotation(self) -> int:
        """Normalize container rotation metadata to one of 0/90/180/270 (clockwise)."""
        rotate_tag = self._stream.metadata.get("rotate")
        if rotate_tag is not None:
            return int(rotate_tag) % 360
        return 0

    @property
    def source_fps(self) -> float:
        return float(self._stream.average_rate)

    @property
    def fps(self) -> float:
        return self.target_fps if self.target_fps else self.source_fps

    def _rotated_raw_size(self) -> tuple[int, int]:
        w, h = self._stream.codec_context.width, self._stream.codec_context.height
        return (h, w) if self._rotation in (90, 270) else (w, h)

    @property
    def width(self) -> int:
        raw_w, raw_h = self._rotated_raw_size()
        return self._scaled_size(raw_w, raw_h)[0]

    @property
    def height(self) -> int:
        raw_w, raw_h = self._rotated_raw_size()
        return self._scaled_size(raw_w, raw_h)[1]

    def _scaled_size(self, raw_w: int, raw_h: int) -> tuple[int, int]:
        if self.max_long_edge is None:
            return raw_w, raw_h
        long_edge = max(raw_w, raw_h)
        if long_edge <= self.max_long_edge:
            return raw_w, raw_h
        scale = self.max_long_edge / long_edge
        return max(1, round(raw_w * scale)), max(1, round(raw_h * scale))

    @property
    def duration_s(self) -> float:
        if self._stream.duration is not None:
            return float(self._stream.duration * self._stream.time_base)
        if self._container.duration is not None:
            return float(self._container.duration / av.time_base)
        raise ValueError(f"could not determine duration for {self.path}")

    def _apply_rotation(self, img: np.ndarray) -> np.ndarray:
        if self._rotation == 90:
            return np.ascontiguousarray(np.rot90(img, k=-1))
        if self._rotation == 180:
            return np.ascontiguousarray(np.rot90(img, k=2))
        if self._rotation == 270:
            return np.ascontiguousarray(np.rot90(img, k=1))
        return img

    def _resize_if_needed(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        target_w, target_h = self._scaled_size(w, h)
        if (target_w, target_h) == (w, h):
            return img
        return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def __iter__(self) -> Iterator[VideoFrame]:
        interval = 1.0 / self.target_fps if self.target_fps else None
        next_sample_t = 0.0
        out_index = 0

        for decode_index, frame in enumerate(self._container.decode(self._stream)):
            if frame.pts is not None:
                t = float(frame.pts * self._stream.time_base)
            else:
                t = decode_index / self.source_fps

            if interval is not None:
                if t + 1e-9 < next_sample_t:
                    continue
                next_sample_t += interval

            img = frame.to_ndarray(format="rgb24")
            img = self._apply_rotation(img)
            img = self._resize_if_needed(img)
            yield VideoFrame(index=out_index, t=t, image=img)
            out_index += 1

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
