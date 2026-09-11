"""Synthetic validation of the support-foot state machine (ARCHITECTURE.md §6.2).

Every AXI-* metric implemented so far depends on this being right, hence
"별도 모듈 + 집중 테스트 대상" (dedicated module, focused test target) in the
architecture doc itself.
"""

import numpy as np

from dvi.geometry.support_foot import detect_support_foot
from tests.fixtures.synthetic.skeletons import build_synthetic_standing_pose

FPS = 30.0


def _pose_with_ankle_heights(t_len: int, left_y: np.ndarray, right_y: np.ndarray, fps: float = FPS):
    pose = build_synthetic_standing_pose(n_frames=t_len, fps=fps)
    joints = pose.joints.copy()
    joints[:, pose.skeleton.joint_index("ankle_l"), 1] += left_y
    joints[:, pose.skeleton.joint_index("ankle_r"), 1] += right_y
    return pose.model_copy(update={"joints": joints})


class TestDoubleSupport:
    def test_both_feet_planted_is_double_throughout(self):
        t_len = 30
        pose = _pose_with_ankle_heights(t_len, np.zeros(t_len), np.zeros(t_len))
        result = detect_support_foot(pose)
        assert np.all(result.state == "double")
        assert np.all(result.left_grounded)
        assert np.all(result.right_grounded)


class TestSingleSupportSwing:
    def test_right_foot_swing_registers_as_left_support_at_the_peak(self):
        t_len = 60
        left_y = np.zeros(t_len)
        right_y = np.zeros(t_len)
        swing_start, swing_end = 10, 50
        arc = 0.15 * np.sin(np.linspace(0, np.pi, swing_end - swing_start))
        right_y[swing_start:swing_end] = arc

        pose = _pose_with_ankle_heights(t_len, left_y, right_y)
        result = detect_support_foot(pose)

        assert result.state[0] == "double"
        assert result.state[swing_start + (swing_end - swing_start) // 2] == "left"
        assert result.state[-1] == "double"
        assert np.all(result.left_grounded)  # left never left the floor

    def test_left_foot_swing_registers_as_right_support(self):
        t_len = 60
        right_y = np.zeros(t_len)
        left_y = np.zeros(t_len)
        swing_start, swing_end = 10, 50
        arc = 0.15 * np.sin(np.linspace(0, np.pi, swing_end - swing_start))
        left_y[swing_start:swing_end] = arc

        pose = _pose_with_ankle_heights(t_len, left_y, right_y)
        result = detect_support_foot(pose)

        assert result.state[swing_start + (swing_end - swing_start) // 2] == "right"


class TestMinDurationFiltering:
    def test_one_frame_graze_during_a_swing_is_not_counted_as_landing(self):
        t_len = 60
        left_y = np.zeros(t_len)
        right_y = np.full(t_len, 0.15)  # right foot held up throughout
        right_y[0:5] = 0.0  # starts grounded
        right_y[30] = 0.0  # a single-frame graze back to the floor mid-swing
        right_y[-5:] = 0.0  # ends grounded

        pose = _pose_with_ankle_heights(t_len, left_y, right_y)
        # min_duration_s=0.05 at 30fps -> needs 2 consecutive grounded frames;
        # the graze is only 1 frame, so it must NOT flip state to "double".
        result = detect_support_foot(pose, min_duration_s=0.05)

        assert result.state[30] == "left"

    def test_sustained_landing_is_not_filtered(self):
        t_len = 60
        left_y = np.zeros(t_len)
        right_y = np.full(t_len, 0.15)
        right_y[0:5] = 0.0
        right_y[25:35] = 0.0  # a real, sustained landing (10 frames)
        right_y[-5:] = 0.0

        pose = _pose_with_ankle_heights(t_len, left_y, right_y)
        result = detect_support_foot(pose, min_duration_s=0.05)

        assert result.state[30] == "double"


class TestFloorEstimate:
    def test_floor_y_tracks_the_planted_foot_height(self):
        t_len = 20
        offset = 1.23  # arbitrary world-frame floor height, not 0
        pose = _pose_with_ankle_heights(t_len, np.full(t_len, offset), np.full(t_len, offset))
        result = detect_support_foot(pose)
        # ankle_l/r sit exactly at the (arbitrary) floor level in this fixture
        assert abs(result.floor_y - pose.joints[0, pose.skeleton.joint_index("ankle_l"), 1]) < 1e-6
