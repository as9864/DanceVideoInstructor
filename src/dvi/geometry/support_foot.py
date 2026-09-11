"""Support-foot state machine (ARCHITECTURE.md §6.2).

"이 판정이 AXI 지표 전부의 기반이므로 별도 모듈 + 집중 테스트 대상" — every
AXI-* metric implemented so far (01/02/04/06) depends on knowing which foot
(if any) is bearing weight at each frame. Hysteresis (separate enter/exit
thresholds) plus a minimum-duration filter keep this from chattering right
at the threshold, the same pattern ARCHITECTURE.md §7.2 uses for Finding
detection.

Floor level isn't known a priori (no fixed "Y=0 is the floor" convention),
so it's estimated per-sequence as a low percentile of ankle heights —
robust to the rare frame where a foot is mid-air below its usual stance
height, unlike taking the strict minimum.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dvi.schemas.frames import Frame
from dvi.schemas.pose import PoseSequence

SupportState = Literal["left", "right", "double", "none"]


@dataclass
class SupportFootResult:
    state: np.ndarray  # (T,) dtype=object, values are SupportState
    left_grounded: np.ndarray  # (T,) bool
    right_grounded: np.ndarray  # (T,) bool
    floor_y: float


def _ground_contact_state_machine(
    height_above_floor: np.ndarray,
    vertical_speed: np.ndarray,
    enter_height: float,
    exit_height: float,
    enter_speed: float,
    exit_speed: float,
) -> np.ndarray:
    """Sequential by nature (state depends on the previous frame) — this is
    the one place in the metrics/geometry code that's a per-frame Python
    loop rather than vectorized (ARCHITECTURE.md §10's "no frame loops"
    targets embarrassingly-parallel metrics, not stateful transitions)."""
    grounded = np.zeros(height_above_floor.shape[0], dtype=bool)
    state = False
    for t in range(height_above_floor.shape[0]):
        if not state:
            if height_above_floor[t] < enter_height and abs(vertical_speed[t]) < enter_speed:
                state = True
        else:
            if height_above_floor[t] > exit_height or abs(vertical_speed[t]) > exit_speed:
                state = False
        grounded[t] = state
    return grounded


def _filter_short_runs(mask: np.ndarray, min_length: int) -> np.ndarray:
    """Flip any contiguous True run shorter than `min_length` frames to False."""
    result = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i < min_length:
                result[i:j] = False
            i = j
        else:
            i += 1
    return result


def detect_support_foot(
    pose: PoseSequence,
    enter_height: float = 0.03,
    exit_height: float = 0.05,
    enter_speed: float = 0.3,
    exit_speed: float = 0.5,
    min_duration_s: float = 0.05,
    floor_percentile: float = 1.0,
) -> SupportFootResult:
    """Heights/speeds are in the same units as `pose.joints` (meters, per
    ARCHITECTURE.md §4.2) — the default thresholds assume that; rescale
    them if a backend's scale convention differs.
    """
    pose.require_frame(Frame.WORLD_G)

    ankle_l_idx = pose.skeleton.joint_index("ankle_l")
    ankle_r_idx = pose.skeleton.joint_index("ankle_r")
    ankle_l_y = pose.joints[:, ankle_l_idx, 1]
    ankle_r_y = pose.joints[:, ankle_r_idx, 1]

    floor_y = float(np.percentile(np.concatenate([ankle_l_y, ankle_r_y]), floor_percentile))
    height_l = ankle_l_y - floor_y
    height_r = ankle_r_y - floor_y

    dt = np.gradient(pose.t)
    vel_l = np.gradient(ankle_l_y) / dt
    vel_r = np.gradient(ankle_r_y) / dt

    grounded_l = _ground_contact_state_machine(height_l, vel_l, enter_height, exit_height, enter_speed, exit_speed)
    grounded_r = _ground_contact_state_machine(height_r, vel_r, enter_height, exit_height, enter_speed, exit_speed)

    mean_fps = 1.0 / np.mean(dt)
    min_frames = max(1, round(min_duration_s * mean_fps))
    grounded_l = _filter_short_runs(grounded_l, min_frames)
    grounded_r = _filter_short_runs(grounded_r, min_frames)

    state = np.full(pose.joints.shape[0], "none", dtype=object)
    state[grounded_l & grounded_r] = "double"
    state[grounded_l & ~grounded_r] = "left"
    state[~grounded_l & grounded_r] = "right"

    return SupportFootResult(state=state, left_grounded=grounded_l, right_grounded=grounded_r, floor_y=floor_y)
