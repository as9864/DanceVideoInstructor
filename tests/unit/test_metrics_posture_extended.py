"""Synthetic/geometric validation of POS-02/04/05a/05b/06 (PROJECT.md §7 V1).

POS-01/03 already have their own test file; these are the "나머지 POS-*"
metrics that came later (ARCHITECTURE.md §13-9).
"""

import numpy as np
import pytest

from dvi.metrics.posture import pos02, pos04, pos05a, pos05b, pos06
from dvi.schemas import Frame, TargetRange
from tests.fixtures.synthetic.skeletons import build_synthetic_standing_pose

DEG_TARGET = TargetRange(unit="deg", max=5.0)
NORM_TARGET = TargetRange(unit="normalized", max=0.12)


class TestPOS02:
    @pytest.mark.parametrize("roll_deg", [0.0, 4.0, -7.0, 15.0])
    def test_reports_exact_input_roll(self, roll_deg):
        pose = build_synthetic_standing_pose(torso_roll_deg=roll_deg)
        result = pos02(pose, DEG_TARGET)
        assert result.series[0] == pytest.approx(roll_deg, abs=1e-6)

    def test_independent_of_sagittal_pitch(self):
        pose = build_synthetic_standing_pose(torso_pitch_deg=10.0, torso_roll_deg=6.0)
        result = pos02(pose, DEG_TARGET)
        assert result.series[0] == pytest.approx(6.0, abs=1e-6)


class TestPOS04:
    def test_reports_exact_forward_head_shift(self):
        pose = build_synthetic_standing_pose(torso_pitch_deg=0.0, torso_length_m=0.5)
        joints = pose.joints.copy()
        joints[:, pose.skeleton.joint_index("head_top"), 2] += 0.06
        shifted = pose.model_copy(update={"joints": joints})

        result = pos04(shifted, NORM_TARGET)
        assert result.series[0] == pytest.approx(0.06 / 0.5, abs=1e-6)

    def test_zero_shift_is_zero(self):
        pose = build_synthetic_standing_pose(torso_pitch_deg=0.0)
        result = pos04(pose, NORM_TARGET)
        assert result.series[0] == pytest.approx(0.0, abs=1e-6)

    def test_requires_world_g_frame(self):
        pose = build_synthetic_standing_pose()
        pose_camera = pose.model_copy(update={"frame": Frame.CAMERA})
        with pytest.raises(ValueError, match="not in required"):
            pos04(pose_camera, NORM_TARGET)


class TestPOS05a:
    def test_shrug_reads_higher_than_baseline(self):
        t_len = 20
        pose = build_synthetic_standing_pose(n_frames=t_len)
        joints = pose.joints.copy()
        shrug = slice(10, 15)
        for side in ("l", "r"):
            idx = pose.skeleton.joint_index(f"shoulder_{side}")
            joints[shrug, idx, 1] += 0.08  # shoulders rise toward the ears
        pose = pose.model_copy(update={"joints": joints})

        result = pos05a(pose, TargetRange(unit="normalized", max=0.10))
        assert result.series[12] > result.series[0]
        assert result.series[12] > 0

    def test_flat_sequence_is_near_zero_baseline(self):
        pose = build_synthetic_standing_pose(n_frames=10)
        result = pos05a(pose, TargetRange(unit="normalized", max=0.10))
        # a p10-baselined, unchanging distance should read ~0 everywhere
        assert result.series == pytest.approx(np.zeros(10), abs=1e-6)


class TestPOS05b:
    def test_reports_exact_shoulder_tilt(self):
        pose = build_synthetic_standing_pose()
        joints = pose.joints.copy()
        delta = 0.02
        joints[:, pose.skeleton.joint_index("shoulder_r"), 1] += delta
        tilted = pose.model_copy(update={"joints": joints})

        result = pos05b(tilted, DEG_TARGET)
        expected = np.degrees(np.arctan2(delta, 0.4))  # default shoulder_width_m = 0.4
        assert result.series[0] == pytest.approx(expected, abs=1e-6)

    def test_level_shoulders_is_zero(self):
        pose = build_synthetic_standing_pose()
        result = pos05b(pose, DEG_TARGET)
        assert result.series[0] == pytest.approx(0.0, abs=1e-6)


class TestPOS06:
    @pytest.mark.parametrize("pitch_deg", [0.0, 12.0, -20.0])
    def test_straight_spine_is_near_zero_at_any_pitch(self, pitch_deg):
        pose = build_synthetic_standing_pose(torso_pitch_deg=pitch_deg, spine_kyphosis_deg=0.0)
        result = pos06(pose, TargetRange(unit="normalized", max=0.05))
        assert result.series[0] == pytest.approx(0.0, abs=1e-6)

    def test_kyphosis_increases_residual(self):
        straight = build_synthetic_standing_pose(spine_kyphosis_deg=0.0)
        collapsed = build_synthetic_standing_pose(spine_kyphosis_deg=20.0)
        r_straight = pos06(straight, TargetRange(unit="normalized", max=0.05)).series[0]
        r_collapsed = pos06(collapsed, TargetRange(unit="normalized", max=0.05)).series[0]
        assert r_collapsed > r_straight > -1e-9
