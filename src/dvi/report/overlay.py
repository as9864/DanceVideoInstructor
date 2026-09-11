"""Skeleton overlay rendering (ARCHITECTURE.md §3 stage 14, M1 milestone).

Draws directly on Pose2D (pixel-space) — this runs before Lift3D/Normalize
exist, so it has no PoseSequence/WORLD_G to work with yet. Only draws what a
backend actually populated: bones/joints missing from `pose.skeleton` or
marked invalid at a given frame are silently skipped, never guessed at.
"""

from pathlib import Path

import cv2
import numpy as np

from dvi.io.video_reader import VideoReader
from dvi.io.video_writer import VideoWriter
from dvi.perception.pose.base import Pose2D
from dvi.schemas.skeleton import CANONICAL_BONES

BONE_COLOR = (0, 255, 0)  # RGB
JOINT_COLOR = (255, 128, 0)


def draw_skeleton(image: np.ndarray, pose: Pose2D, t_index: int) -> np.ndarray:
    """Draw one frame's skeleton onto `image` (RGB, HxWx3) in place; returns it."""
    if t_index >= pose.joints_px.shape[0]:
        return image

    joint_index = {name: i for i, name in enumerate(pose.skeleton.joint_names)}

    for name_a, name_b in CANONICAL_BONES:
        if name_a not in joint_index or name_b not in joint_index:
            continue
        ia, ib = joint_index[name_a], joint_index[name_b]
        if not (pose.valid[t_index, ia] and pose.valid[t_index, ib]):
            continue
        pa = tuple(int(round(c)) for c in pose.joints_px[t_index, ia])
        pb = tuple(int(round(c)) for c in pose.joints_px[t_index, ib])
        cv2.line(image, pa, pb, color=BONE_COLOR, thickness=2, lineType=cv2.LINE_AA)

    for name, idx in joint_index.items():
        if not pose.valid[t_index, idx]:
            continue
        p = tuple(int(round(c)) for c in pose.joints_px[t_index, idx])
        cv2.circle(image, p, radius=3, color=JOINT_COLOR, thickness=-1, lineType=cv2.LINE_AA)

    return image


def _rescale_to(pose: Pose2D, width: int, height: int) -> Pose2D:
    """Rescale joints_px if `pose` was inferred at a different resolution than
    what's being drawn on (e.g. inference ran on a downscaled/max_long_edge
    copy of the frames). A no-op when sizes already match."""
    src_w, src_h = pose.image_size
    if (src_w, src_h) == (width, height):
        return pose
    scale = np.array([width / src_w, height / src_h])
    return Pose2D(
        subject_id=pose.subject_id,
        t=pose.t,
        joints_px=pose.joints_px * scale,
        confidence=pose.confidence,
        valid=pose.valid,
        skeleton=pose.skeleton,
        image_size=(width, height),
    )


def render_overlay(video_path: Path, pose: Pose2D, out_path: Path) -> Path:
    """Render `pose` as a skeleton overlay on top of `video_path`, written to `out_path`.

    `pose` is rescaled to the reader's resolution if it was inferred at a
    different one (e.g. the backend ran on a max_long_edge-downscaled copy
    of the frames) — otherwise the skeleton would drift off the body.
    """
    reader = VideoReader(video_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pose = _rescale_to(pose, reader.width, reader.height)

    with VideoWriter(out_path, fps=reader.fps, width=reader.width, height=reader.height) as writer:
        for frame in reader:
            img = frame.image.copy()
            draw_skeleton(img, pose, frame.index)
            writer.write(img)

    return out_path
