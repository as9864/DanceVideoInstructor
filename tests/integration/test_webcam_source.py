"""Frame sources for `dvi live` (io/webcam.py) — the file-replay path and
source-spec parsing. A real camera isn't available in CI; LatestFrameCapture
is exercised by reading the same clip through cv2.VideoCapture."""

from pathlib import Path

import numpy as np
import pytest

from dvi.io.webcam import FileFrameSource, LatestFrameCapture, open_source, parse_source
from dvi.io.video_writer import VideoWriter

W, H, N = 64, 48, 12


@pytest.fixture
def clip(tmp_path: Path) -> Path:
    path = tmp_path / "clip.mp4"
    with VideoWriter(path, fps=10, width=W, height=H) as writer:
        for i in range(N):
            img = np.zeros((H, W, 3), dtype=np.uint8)
            img[:, : W // 4] = 255  # bright band on the left, to check mirroring
            writer.write(img)
    return path


def test_parse_source():
    assert parse_source("0") == 0
    assert parse_source("http://192.168.0.5:4747/video") == "http://192.168.0.5:4747/video"
    assert parse_source("clip.mp4") == Path("clip.mp4")


def test_file_source_reads_every_frame_in_order(clip):
    src = open_source(str(clip))
    assert isinstance(src, FileFrameSource) and not src.mirror
    frames = []
    while (got := src.read()) is not None:
        frames.append(got)
    src.close()
    assert len(frames) == N
    ts = [t for _, t in frames]
    assert ts == sorted(ts)
    assert frames[0][0][:, :4].mean() > 200  # band stays on the left


def test_file_source_mirror(clip):
    src = open_source(str(clip), mirror=True)
    img, _ = src.read()
    src.close()
    assert img[:, -4:].mean() > 200 and img[:, :4].mean() < 50


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        open_source("does/not/exist.mp4")


def test_latest_frame_capture_delivers_frames(clip):
    cap = LatestFrameCapture(str(clip), max_long_edge=32)
    got = cap.read()
    assert got is not None
    img, t = got
    assert max(img.shape[:2]) <= 32 and t >= 0.0
    while cap.read(timeout_s=1.0) is not None:  # drains until the clip ends
        pass
    cap.close()
