"""Synthetic/geometric validation of POS-01/POS-03 (PROJECT.md §7 V1).

Feed a skeleton with a known, exact angle; assert the metric reports that
angle. The pitch-vs-kyphosis independence tests are the actual point of
these metrics existing as two separate numbers (ARCHITECTURE.md §6.1).
"""

import numpy as np
import pytest

from dvi.metrics.posture import pos01, pos03
from dvi.schemas import Frame, TargetRange
from tests.fixtures.synthetic.skeletons import build_synthetic_standing_pose

SALON_POS01_TARGET = TargetRange(unit="deg", min=0.0, max=5.0, tolerance=2.0)
SALON_POS03_TARGET = TargetRange(unit="deg", max=8.0)


class TestPOS01:
    @pytest.mark.parametrize("pitch_deg", [0.0, 5.0, -5.0, 12.0, -30.0])
    def test_reports_exact_input_pitch(self, pitch_deg):
        pose = build_synthetic_standing_pose(torso_pitch_deg=pitch_deg)
        result = pos01(pose, SALON_POS01_TARGET)
        assert result.series[0] == pytest.approx(pitch_deg, abs=1e-6)
        assert result.coverage == 1.0

    @pytest.mark.parametrize("roll_deg", [0.0, 15.0, -20.0])
    def test_independent_of_coronal_roll(self, roll_deg):
        pose = build_synthetic_standing_pose(torso_pitch_deg=7.0, torso_roll_deg=roll_deg)
        result = pos01(pose, SALON_POS01_TARGET)
        assert result.series[0] == pytest.approx(7.0, abs=1e-6)

    def test_independent_of_spine_kyphosis(self):
        # POS-01 uses hip_mid->neck directly, never touching spine_mid.
        straight = build_synthetic_standing_pose(torso_pitch_deg=6.0, spine_kyphosis_deg=0.0)
        collapsed = build_synthetic_standing_pose(torso_pitch_deg=6.0, spine_kyphosis_deg=25.0)
        assert pos01(straight, SALON_POS01_TARGET).series[0] == pytest.approx(
            pos01(collapsed, SALON_POS01_TARGET).series[0], abs=1e-6
        )

    def test_requires_world_g_frame(self):
        pose = build_synthetic_standing_pose()
        pose_camera = pose.model_copy(update={"frame": Frame.CAMERA})
        with pytest.raises(ValueError, match="not in required"):
            pos01(pose_camera, SALON_POS01_TARGET)


class TestPOS03:
    @pytest.mark.parametrize("kyphosis_deg", [0.0, 8.0, 20.0, 45.0])
    def test_reports_exact_input_kyphosis(self, kyphosis_deg):
        pose = build_synthetic_standing_pose(spine_kyphosis_deg=kyphosis_deg)
        result = pos03(pose, SALON_POS03_TARGET)
        assert result.series[0] == pytest.approx(kyphosis_deg, abs=1e-3)

    def test_independent_of_torso_pitch(self):
        # Same kyphosis, wildly different overall lean — POS-03 must not move.
        upright = build_synthetic_standing_pose(torso_pitch_deg=0.0, spine_kyphosis_deg=15.0)
        leaning = build_synthetic_standing_pose(torso_pitch_deg=20.0, spine_kyphosis_deg=15.0)
        assert pos03(upright, SALON_POS03_TARGET).series[0] == pytest.approx(
            pos03(leaning, SALON_POS03_TARGET).series[0], abs=1e-3
        )

    def test_straight_spine_is_zero(self):
        pose = build_synthetic_standing_pose(spine_kyphosis_deg=0.0)
        result = pos03(pose, SALON_POS03_TARGET)
        assert result.series[0] == pytest.approx(0.0, abs=1e-6)


class TestCoverage:
    def test_invalid_frames_excluded_from_aggregate_but_kept_in_series(self):
        pose = build_synthetic_standing_pose(torso_pitch_deg=5.0, n_frames=4)
        valid = pose.valid.copy()
        valid[2, :] = False  # simulate one occluded frame
        pose = pose.model_copy(update={"valid": valid})

        result = pos01(pose, SALON_POS01_TARGET)
        assert result.series.shape == (4,)  # series keeps its shape...
        assert result.coverage == pytest.approx(0.75)  # ...but coverage reflects the drop
        assert result.aggregate["mean"] == pytest.approx(5.0, abs=1e-6)  # invalid frame excluded from stats


def test_pos01_and_pos03_are_independent_axes():
    """The headline claim of ARCHITECTURE.md §6.1: a forward-leaning-but-straight
    torso and an upright-but-collapsed torso must score oppositely on the two metrics."""
    leaning_straight = build_synthetic_standing_pose(torso_pitch_deg=10.0, spine_kyphosis_deg=0.0)
    upright_collapsed = build_synthetic_standing_pose(torso_pitch_deg=0.0, spine_kyphosis_deg=10.0)

    assert pos01(leaning_straight, SALON_POS01_TARGET).series[0] == pytest.approx(10.0, abs=1e-6)
    assert pos03(leaning_straight, SALON_POS03_TARGET).series[0] == pytest.approx(0.0, abs=1e-6)

    assert pos01(upright_collapsed, SALON_POS01_TARGET).series[0] == pytest.approx(0.0, abs=1e-6)
    assert pos03(upright_collapsed, SALON_POS03_TARGET).series[0] == pytest.approx(10.0, abs=1e-3)
