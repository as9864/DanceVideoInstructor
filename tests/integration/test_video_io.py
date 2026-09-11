"""Encode/decode roundtrip for io/video_reader.py + io/video_writer.py.

Generates a tiny synthetic clip with PyAV rather than shipping a binary
fixture — keeps the repo free of video assets for a check this cheap.
"""

from pathlib import Path

import av
import numpy as np
import pytest

from dvi.io.video_reader import VideoReader
from dvi.io.video_writer import VideoWriter

WIDTH, HEIGHT, FPS, N_FRAMES = 64, 48, 10, 20


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width = WIDTH
    stream.height = HEIGHT
    stream.pix_fmt = "yuv420p"

    for i in range(N_FRAMES):
        img = np.full((HEIGHT, WIDTH, 3), fill_value=(i * 12) % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


class TestVideoReader:
    def test_reads_all_frames_with_correct_size_and_fps(self, tiny_video):
        reader = VideoReader(tiny_video)
        frames = list(reader)

        assert len(frames) == N_FRAMES
        assert reader.width == WIDTH
        assert reader.height == HEIGHT
        assert abs(reader.fps - FPS) < 0.5
        assert frames[0].image.shape == (HEIGHT, WIDTH, 3)
        # timestamps must be strictly increasing — every downstream series is indexed by them
        ts = [f.t for f in frames]
        assert all(b > a for a, b in zip(ts, ts[1:]))

    def test_resizes_to_max_long_edge(self, tiny_video):
        reader = VideoReader(tiny_video, max_long_edge=32)
        frame = next(iter(reader))
        assert max(frame.image.shape[:2]) <= 32
        assert reader.width <= 32 and reader.height <= 32

    def test_resized_dimensions_are_always_even(self, tiny_video):
        # yuv420p (io/video_writer.py's encode target) can't handle odd
        # width/height; a naive round() on the resize scale factor can land
        # on an odd number, which then crashes libx264 downstream.
        reader = VideoReader(tiny_video, max_long_edge=33)  # picked to force an odd rounding
        assert reader.width % 2 == 0
        assert reader.height % 2 == 0
        frame = next(iter(reader))
        assert frame.image.shape[0] % 2 == 0
        assert frame.image.shape[1] % 2 == 0

    def test_target_fps_downsamples(self, tiny_video):
        # source is 10fps/20 frames (2.0s) -> 5fps should yield ~10 frames, not 20
        reader = VideoReader(tiny_video, target_fps=5.0)
        frames = list(reader)
        assert len(frames) < N_FRAMES

    def test_duration_is_positive(self, tiny_video):
        reader = VideoReader(tiny_video)
        assert reader.duration_s > 0


class TestVideoWriter:
    def test_write_then_read_back_same_frame_count(self, tmp_path):
        out_path = tmp_path / "out.mp4"
        with VideoWriter(out_path, fps=FPS, width=WIDTH, height=HEIGHT) as writer:
            for i in range(N_FRAMES):
                img = np.full((HEIGHT, WIDTH, 3), fill_value=i % 256, dtype=np.uint8)
                writer.write(img)

        assert out_path.exists()
        reader = VideoReader(out_path)
        frames = list(reader)
        assert len(frames) == N_FRAMES
