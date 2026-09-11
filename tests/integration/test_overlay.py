"""render_overlay must rescale joints if inference ran at a different
resolution than the video it draws on (e.g. a max_long_edge-downscaled
copy of the frames) — otherwise the skeleton drifts off the body.
"""

import av
import numpy as np
import pytest

from dvi.io.video_reader import VideoReader
from dvi.perception.pose.base import Pose2D
from dvi.perception.pose.skeleton import COCO17_SKELETON_SPEC
from dvi.report.overlay import render_overlay

WIDTH, HEIGHT, FPS, N_FRAMES = 64, 48, 10, 3
BACKGROUND = 20  # flat gray background pixel value, all channels

# libx264 is lossy, so exact RGB tuples don't survive an encode/decode
# roundtrip — check "clearly not background" instead of exact color match.
def _is_marked(pixel) -> bool:
    return int(np.abs(pixel.astype(int) - BACKGROUND).sum()) > 100


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "tiny.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
    for _ in range(N_FRAMES):
        img = np.full((HEIGHT, WIDTH, 3), 20, dtype=np.uint8)  # dark background
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def _single_joint_pose(image_size: tuple[int, int], px: tuple[float, float]) -> Pose2D:
    j = COCO17_SKELETON_SPEC.n_joints
    nose_idx = COCO17_SKELETON_SPEC.joint_index("nose")
    joints_px = np.zeros((N_FRAMES, j, 2))
    joints_px[:, nose_idx] = px
    confidence = np.zeros((N_FRAMES, j))
    confidence[:, nose_idx] = 1.0
    valid = confidence > 0.5
    return Pose2D(
        subject_id="s",
        t=np.arange(N_FRAMES, dtype=float) / FPS,
        joints_px=joints_px,
        confidence=confidence,
        valid=valid,
        skeleton=COCO17_SKELETON_SPEC,
        image_size=image_size,
    )


def test_matching_resolution_draws_at_given_pixel(tmp_path, tiny_video):
    pose = _single_joint_pose(image_size=(WIDTH, HEIGHT), px=(48.0, 36.0))  # 3/4 across, 3/4 down
    out_path = tmp_path / "overlay.mp4"
    render_overlay(tiny_video, pose, out_path)

    frame = next(iter(VideoReader(out_path)))
    assert _is_marked(frame.image[36, 48])


def test_mismatched_resolution_is_rescaled_not_left_in_place(tmp_path, tiny_video):
    # Pose inferred at half resolution (32x24); the joint sits at its center (16, 12).
    # Drawn on the full 64x48 video, it must land at the *scaled* center (32, 24), not (16, 12).
    pose = _single_joint_pose(image_size=(WIDTH // 2, HEIGHT // 2), px=(16.0, 12.0))
    out_path = tmp_path / "overlay.mp4"
    render_overlay(tiny_video, pose, out_path)

    frame = next(iter(VideoReader(out_path)))
    assert _is_marked(frame.image[24, 32])
    assert not _is_marked(frame.image[12, 16])
