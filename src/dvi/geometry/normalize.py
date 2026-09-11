"""BODY frame construction (ARCHITECTURE.md §4.1, §5.5).

Pelvis-origin, hip-line yaw-aligned — rotation-invariant metrics (POS-04,
POS-05a) need this so a dancer facing any direction gets the same reading.
"""

import numpy as np

from dvi.schemas.frames import Frame
from dvi.schemas.pose import PoseSequence


def to_body_frame(pose: PoseSequence) -> PoseSequence:
    """WORLD_G -> BODY: recenter on the pelvis, then rotate about the world
    Y-axis so the hip line (hip_l -> hip_r) points along +X at every frame.

    Only a yaw rotation is applied — pitch/roll (established by gravity
    alignment) are left untouched, since BODY only adds a per-frame facing
    reference on top of WORLD_G, it doesn't re-solve verticality.
    """
    pose.require_frame(Frame.WORLD_G)

    pelvis_idx = pose.skeleton.joint_index("pelvis")
    hip_l_idx = pose.skeleton.joint_index("hip_l")
    hip_r_idx = pose.skeleton.joint_index("hip_r")

    pelvis = pose.joints[:, pelvis_idx]
    hip_line = pose.joints[:, hip_r_idx] - pose.joints[:, hip_l_idx]  # (T, 3)

    yaw = np.arctan2(hip_line[:, 2], hip_line[:, 0])  # angle from +X in the XZ plane
    cos_yaw, sin_yaw = np.cos(-yaw), np.sin(-yaw)  # rotate by -yaw to bring hip line onto +X

    centered = pose.joints - pelvis[:, None, :]
    x, z = centered[..., 0], centered[..., 2]
    rotated = centered.copy()
    rotated[..., 0] = x * cos_yaw[:, None] - z * sin_yaw[:, None]
    rotated[..., 2] = x * sin_yaw[:, None] + z * cos_yaw[:, None]

    return pose.model_copy(update={"joints": rotated, "frame": Frame.BODY})
