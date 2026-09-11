"""AXI-* axis/balance metrics (ARCHITECTURE.md §6.2).

Pure functions: (PoseSequence, SupportFootResult, TargetRange) -> MetricResult.
All four here are restricted to single-support frames (state == "left" or
"right") — during double support there's no single well-defined "the
support leg/foot" to measure against, so those frames are marked invalid
rather than guessed at.

AXI-03 (weight-transfer clarity) and AXI-05 (ground-contact quality) are
NOT implemented here: both need event detection (transition boundaries,
footfall ordering) beyond what a per-frame series over the support-foot
state machine gives you — that's a Segment-stage-shaped problem
(ARCHITECTURE.md §3 stage 11), not yet built. Left for a follow-up pass.
"""

import numpy as np

from dvi.geometry.com import estimate_com
from dvi.geometry.support_foot import SupportFootResult
from dvi.schemas.frames import Frame
from dvi.schemas.metrics import MetricResult, TargetRange
from dvi.schemas.pose import PoseSequence


def _joint(pose: PoseSequence, name: str) -> np.ndarray:
    return pose.joints[:, pose.skeleton.joint_index(name)]


def _select_support_side(is_left: np.ndarray, left_value: np.ndarray, right_value: np.ndarray) -> np.ndarray:
    mask = is_left[:, None] if left_value.ndim > 1 else is_left
    return np.where(mask, left_value, right_value)


def _support_side_joint_valid(pose: PoseSequence, is_left: np.ndarray, joint_base: str) -> np.ndarray:
    valid_l = pose.valid[:, pose.skeleton.joint_index(f"{joint_base}_l")]
    valid_r = pose.valid[:, pose.skeleton.joint_index(f"{joint_base}_r")]
    return np.where(is_left, valid_l, valid_r)


def _height_or_torso_scale(pose: PoseSequence) -> float:
    scale = pose.scale.height_estimate_m if pose.scale.height_estimate_m is not None else pose.scale.torso_length_m
    if scale is None:
        raise ValueError("this metric requires BodyScale.height_estimate_m or torso_length_m for normalization")
    return scale


def compute_axi01_series(pose: PoseSequence, support: SupportFootResult) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal COM-to-support-foot offset, normalized by height
    (ARCHITECTURE.md §6.2 AXI-01). Single-support frames only.
    """
    pose.require_frame(Frame.WORLD_G)
    com, com_valid = estimate_com(pose)
    is_left = support.state == "left"
    single_support = is_left | (support.state == "right")

    support_point = _select_support_side(is_left, _joint(pose, "ankle_l"), _joint(pose, "ankle_r"))
    offset = np.linalg.norm(com[:, [0, 2]] - support_point[:, [0, 2]], axis=-1)
    series = offset / _height_or_torso_scale(pose)

    ankle_valid = _support_side_joint_valid(pose, is_left, "ankle")
    valid = single_support & com_valid & ankle_valid
    return series, valid


def compute_axi02_series(pose: PoseSequence, support: SupportFootResult) -> tuple[np.ndarray, np.ndarray]:
    """Instantaneous horizontal-COM jerk magnitude during single support
    (ARCHITECTURE.md §6.2 AXI-02).

    Simplifies the spec's per-stance-window "표준편차 및 저크 RMS" into a
    single per-frame jerk-magnitude series so it fits the same (T,)
    MetricResult contract as every other metric — MetricResult.aggregate's
    mean/p95 over the single-support frames stands in for a windowed
    statistic. Revisit as a proper per-stance aggregate once a Segment
    stage exists to delimit stance windows explicitly.

    Position is normalized by height/torso length before differentiating
    (same scale reference as AXI-01), matching the "normalized" unit in
    configs/styles/*.yaml's AXI-02 target — a dancer's absolute jerk scales
    with limb length, so comparing raw meters across different body sizes
    wouldn't be meaningful.
    """
    pose.require_frame(Frame.WORLD_G)
    com, com_valid = estimate_com(pose)
    single_support = (support.state == "left") | (support.state == "right")

    dt = np.gradient(pose.t)
    horizontal = com[:, [0, 2]] / _height_or_torso_scale(pose)
    velocity = np.gradient(horizontal, axis=0) / dt[:, None]
    acceleration = np.gradient(velocity, axis=0) / dt[:, None]
    jerk = np.gradient(acceleration, axis=0) / dt[:, None]
    jerk_magnitude = np.linalg.norm(jerk, axis=-1)

    valid = single_support & com_valid
    return jerk_magnitude, valid


def compute_axi04_series(pose: PoseSequence, support: SupportFootResult) -> tuple[np.ndarray, np.ndarray]:
    """Support-knee flexion angle, hip-knee-ankle, 180°=fully extended
    (ARCHITECTURE.md §6.2 AXI-04). Single-support frames only.
    """
    pose.require_frame(Frame.WORLD_G)
    is_left = support.state == "left"
    single_support = is_left | (support.state == "right")

    hip = _select_support_side(is_left, _joint(pose, "hip_l"), _joint(pose, "hip_r"))
    knee = _select_support_side(is_left, _joint(pose, "knee_l"), _joint(pose, "knee_r"))
    ankle = _select_support_side(is_left, _joint(pose, "ankle_l"), _joint(pose, "ankle_r"))

    v1 = hip - knee
    v2 = ankle - knee
    cos_theta = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    joint_valid = (
        _support_side_joint_valid(pose, is_left, "hip")
        & _support_side_joint_valid(pose, is_left, "knee")
        & _support_side_joint_valid(pose, is_left, "ankle")
    )
    valid = single_support & joint_valid
    return angle_deg, valid


def compute_axi06_series(pose: PoseSequence, support: SupportFootResult) -> tuple[np.ndarray, np.ndarray]:
    """Angle between (COM - support ankle) and world-up (ARCHITECTURE.md §6.2 AXI-06).
    Single-support frames only.
    """
    pose.require_frame(Frame.WORLD_G)
    com, com_valid = estimate_com(pose)
    is_left = support.state == "left"
    single_support = is_left | (support.state == "right")

    ankle = _select_support_side(is_left, _joint(pose, "ankle_l"), _joint(pose, "ankle_r"))
    v = com - ankle
    norm = np.linalg.norm(v, axis=-1)
    cos_theta = np.divide(v[:, 1], norm, out=np.full_like(norm, np.nan), where=norm > 0)
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    ankle_valid = _support_side_joint_valid(pose, is_left, "ankle")
    valid = single_support & com_valid & ankle_valid & (norm > 0)
    return angle_deg, valid


def _to_metric_result(
    metric_id: str, series: np.ndarray, valid: np.ndarray, target: TargetRange, causal: bool = True
) -> MetricResult:
    coverage = float(np.mean(valid)) if valid.size else 0.0
    observed = series[valid]
    aggregate = {
        "mean": float(np.mean(observed)) if observed.size else float("nan"),
        "p50": float(np.percentile(observed, 50)) if observed.size else float("nan"),
        "p95": float(np.percentile(observed, 95)) if observed.size else float("nan"),
    }
    return MetricResult(
        metric_id=metric_id, series=series, valid=valid, coverage=coverage, target=target,
        aggregate=aggregate, causal=causal,
    )


def axi01(pose: PoseSequence, support: SupportFootResult, target: TargetRange) -> MetricResult:
    series, valid = compute_axi01_series(pose, support)
    return _to_metric_result("AXI-01", series, valid, target)


def axi02(pose: PoseSequence, support: SupportFootResult, target: TargetRange) -> MetricResult:
    # ARCHITECTURE.md §6.2: causal "✔(윈도우)" — a short local window (jerk
    # needs a few neighboring frames), not the full non-causal lookahead
    # AXI-03/CPL-04 require.
    series, valid = compute_axi02_series(pose, support)
    return _to_metric_result("AXI-02", series, valid, target, causal=True)


def axi04(pose: PoseSequence, support: SupportFootResult, target: TargetRange) -> MetricResult:
    series, valid = compute_axi04_series(pose, support)
    return _to_metric_result("AXI-04", series, valid, target)


def axi06(pose: PoseSequence, support: SupportFootResult, target: TargetRange) -> MetricResult:
    series, valid = compute_axi06_series(pose, support)
    return _to_metric_result("AXI-06", series, valid, target)
