"""Pose2D contract and PoseBackend adapter protocol (ARCHITECTURE.md §3 stage 4, §5.1).

Pose2D is a perception-layer intermediate (Detect → Track → Pose2D → Lift3D
→ ... → PoseSequence), not the cross-layer contract — that's PoseSequence
(dvi.schemas.pose), formed only after Normalize. Keeping this one separate
means adding a Track/multi-person dimension later doesn't touch schemas/.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from dvi.io.video_reader import VideoFrame
from dvi.schemas.skeleton import SkeletonSpec


@dataclass
class Pose2D:
    subject_id: str
    t: np.ndarray  # (T,) seconds
    joints_px: np.ndarray  # (T, J, 2) pixel coords, in `skeleton` joint order
    confidence: np.ndarray  # (T, J) in [0, 1]
    valid: np.ndarray  # (T, J) bool — confidence-gated
    skeleton: SkeletonSpec  # subset of DVI-Canonical this backend actually populates
    image_size: tuple[int, int]  # (width, height), for overlay/denormalization


@runtime_checkable
class PoseBackend(Protocol):
    name: str
    provides_3d: bool
    provides_world_frame: bool
    skeleton: SkeletonSpec

    def infer(self, frames: Sequence[VideoFrame]) -> Pose2D: ...
