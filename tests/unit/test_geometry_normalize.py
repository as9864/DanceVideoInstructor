"""BODY frame construction (PROJECT.md §7 V1 + §12 rotation-invariance property).

The headline property: yaw-rotating a pose in WORLD_G before conversion
must not change the resulting BODY-frame geometry — a dancer facing any
direction gets the same reading.
"""

import numpy as np
import pytest

from dvi.geometry.normalize import to_body_frame
from dvi.schemas import Frame
from tests.fixtures.synthetic.skeletons import build_synthetic_standing_pose


def _yaw_rotate_pose(pose, yaw_deg: float):
    yaw = np.radians(yaw_deg)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    joints = pose.joints.copy()
    x, z = joints[..., 0].copy(), joints[..., 2].copy()  # copy: joints[...,0] is a view, not independent data
    joints[..., 0] = x * cos_y - z * sin_y
    joints[..., 2] = x * sin_y + z * cos_y
    return pose.model_copy(update={"joints": joints})


class TestToBodyFrame:
    def test_recenters_on_pelvis(self):
        pose = build_synthetic_standing_pose()
        body = to_body_frame(pose)
        pelvis = body.joints[:, body.skeleton.joint_index("pelvis")]
        assert pelvis[0] == pytest.approx(np.zeros(3), abs=1e-9)

    def test_hip_line_lands_on_plus_x(self):
        pose = build_synthetic_standing_pose()
        body = to_body_frame(pose)
        hip_l = body.joints[:, body.skeleton.joint_index("hip_l")]
        hip_r = body.joints[:, body.skeleton.joint_index("hip_r")]
        hip_line = hip_r[0] - hip_l[0]
        assert hip_line[2] == pytest.approx(0.0, abs=1e-9)  # no Z component left
        assert hip_line[0] > 0  # hip_r is in the +X direction

    def test_sets_frame_to_body(self):
        pose = build_synthetic_standing_pose()
        body = to_body_frame(pose)
        assert body.frame == Frame.BODY

    @pytest.mark.parametrize("yaw_deg", [0.0, 40.0, -95.0, 170.0])
    def test_invariant_to_original_yaw(self, yaw_deg):
        """The core property (ARCHITECTURE.md §12): rotate the dancer to face
        any direction in WORLD_G, BODY-frame geometry must come out identical."""
        pose = build_synthetic_standing_pose(torso_pitch_deg=8.0)
        rotated = _yaw_rotate_pose(pose, yaw_deg)

        body_from_original = to_body_frame(pose)
        body_from_rotated = to_body_frame(rotated)

        assert body_from_rotated.joints == pytest.approx(body_from_original.joints, abs=1e-6)

    def test_requires_world_g_frame(self):
        pose = build_synthetic_standing_pose()
        pose_camera = pose.model_copy(update={"frame": Frame.CAMERA})
        with pytest.raises(ValueError, match="not in required"):
            to_body_frame(pose_camera)
