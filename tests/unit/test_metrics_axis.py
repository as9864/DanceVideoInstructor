"""Synthetic/geometric validation of AXI-01/02/04/06 (PROJECT.md §7 V1).

AXI-04 (a plain 3-point angle) gets an exact hand-derived ground truth.
AXI-01/02/06 (which depend on the segment-weighted COM, already covered by
tests/unit/test_geometry_com.py) are checked by recomputing the same
estimate_com() call independently in the test — this validates that the
metric *composes* COM + support-ankle + scale correctly without
re-deriving the COM formula a second time by hand.
"""

import numpy as np
import pytest

from dvi.geometry.com import estimate_com
from dvi.geometry.support_foot import SupportFootResult
from dvi.metrics.axis import axi01, axi02, axi04, axi06
from dvi.schemas import TargetRange
from tests.fixtures.synthetic.skeletons import build_synthetic_standing_pose

NORM_TARGET = TargetRange(unit="normalized", max=0.05)
DEG_TARGET = TargetRange(unit="deg", min=165.0, max=178.0)
VERTICALITY_TARGET = TargetRange(unit="deg", max=4.0)


def _support(state: str, t_len: int = 1) -> SupportFootResult:
    states = np.full(t_len, state, dtype=object)
    return SupportFootResult(
        state=states,
        left_grounded=states == "left",
        right_grounded=states == "right",
        floor_y=0.0,
    )


def _collapse_all_joints_to(pose, x: float, z: float):
    joints = pose.joints.copy()
    joints[..., 0] = x
    joints[..., 2] = z
    return pose.model_copy(update={"joints": joints})


class TestAXI01:
    def test_matches_independently_recomputed_com_and_ankle(self):
        pose = build_synthetic_standing_pose()
        support = _support("left")
        result = axi01(pose, support, NORM_TARGET)

        com, _ = estimate_com(pose)
        ankle_l = pose.joints[:, pose.skeleton.joint_index("ankle_l")]
        expected = np.linalg.norm(com[:, [0, 2]] - ankle_l[:, [0, 2]], axis=-1) / pose.scale.torso_length_m
        assert result.series[0] == pytest.approx(expected[0])

    def test_zero_offset_when_com_directly_above_support_ankle(self):
        pose = _collapse_all_joints_to(build_synthetic_standing_pose(), x=0.2, z=0.3)
        result = axi01(pose, _support("left"), NORM_TARGET)
        assert result.series[0] == pytest.approx(0.0, abs=1e-9)

    def test_double_support_frames_are_invalid(self):
        pose = build_synthetic_standing_pose()
        result = axi01(pose, _support("double"), NORM_TARGET)
        assert result.coverage == 0.0

    def test_no_support_frames_are_invalid(self):
        pose = build_synthetic_standing_pose()
        result = axi01(pose, _support("none"), NORM_TARGET)
        assert result.coverage == 0.0


class TestAXI04:
    def test_straight_leg_is_180_degrees(self):
        # default fixture builds a fully-extended leg (hip/knee/ankle colinear)
        pose = build_synthetic_standing_pose()
        result = axi04(pose, _support("left"), DEG_TARGET)
        assert result.series[0] == pytest.approx(180.0, abs=1e-6)

    def test_known_knee_bend_matches_hand_derivation(self):
        bend_deg = 10.0
        pose = build_synthetic_standing_pose()
        joints = pose.joints.copy()
        knee_idx = pose.skeleton.joint_index("knee_l")
        ankle_idx = pose.skeleton.joint_index("ankle_l")
        knee = joints[0, knee_idx]
        bend_rad = np.radians(bend_deg)
        # ankle placed so the hip-knee-ankle angle is exactly 180 - bend_deg
        joints[0, ankle_idx] = knee + 0.45 * np.array([0.0, -np.cos(bend_rad), np.sin(bend_rad)])
        pose = pose.model_copy(update={"joints": joints})

        result = axi04(pose, _support("left"), DEG_TARGET)
        assert result.series[0] == pytest.approx(180.0 - bend_deg, abs=1e-6)

    def test_uses_right_leg_when_right_is_support(self):
        pose = build_synthetic_standing_pose()
        result = axi04(pose, _support("right"), DEG_TARGET)
        assert result.series[0] == pytest.approx(180.0, abs=1e-6)

    def test_double_support_is_invalid(self):
        pose = build_synthetic_standing_pose()
        result = axi04(pose, _support("double"), DEG_TARGET)
        assert result.coverage == 0.0


class TestAXI06:
    def test_zero_angle_when_com_directly_above_ankle(self):
        pose = _collapse_all_joints_to(build_synthetic_standing_pose(), x=0.0, z=0.0)
        result = axi06(pose, _support("left"), VERTICALITY_TARGET)
        assert result.series[0] == pytest.approx(0.0, abs=1e-6)

    def test_matches_independently_recomputed_com_and_ankle(self):
        pose = build_synthetic_standing_pose()
        support = _support("left")
        result = axi06(pose, support, VERTICALITY_TARGET)

        com, _ = estimate_com(pose)
        ankle_l = pose.joints[:, pose.skeleton.joint_index("ankle_l")]
        v = com[0] - ankle_l[0]
        expected = np.degrees(np.arccos(np.clip(v[1] / np.linalg.norm(v), -1.0, 1.0)))
        assert result.series[0] == pytest.approx(expected)


class TestAXI02:
    def test_stationary_com_has_zero_jerk(self):
        t_len = 10
        pose = build_synthetic_standing_pose(n_frames=t_len)  # identical frame repeated -> static COM
        support = _support("left", t_len)
        result = axi02(pose, support, TargetRange(unit="normalized", max=0.02))
        # central-difference jerk is undefined right at the array edges only
        # in the sense of one-sided differencing; a truly static series is
        # zero everywhere regardless.
        assert result.series == pytest.approx(np.zeros(t_len), abs=1e-9)

    def test_only_single_support_frames_are_valid(self):
        t_len = 10
        pose = build_synthetic_standing_pose(n_frames=t_len)
        states = np.array(["left"] * 5 + ["double"] * 5, dtype=object)
        support = SupportFootResult(
            state=states, left_grounded=states != "none", right_grounded=states == "double", floor_y=0.0
        )
        result = axi02(pose, support, TargetRange(unit="normalized", max=0.02))
        assert result.valid[:5].all()
        assert not result.valid[5:].any()

    def test_causal_flag_is_true(self):
        pose = build_synthetic_standing_pose(n_frames=5)
        result = axi02(pose, _support("left", 5), TargetRange(unit="normalized", max=0.02))
        assert result.causal is True
