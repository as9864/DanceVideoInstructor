"""POS-* posture metrics (ARCHITECTURE.md §6.1).

Pure functions: (PoseSequence, TargetRange) -> MetricResult. No I/O, no
model dependencies (design principle P3) — every metric here is testable
against synthetic geometry alone (PROJECT.md §7 V1).

POS-01 vs POS-03 is the load-bearing distinction the architecture calls out:
POS-01 is the *whole* torso leaning as one rigid unit (normal in milonguero),
POS-03 is the spine *kinking* partway up (always a fault). They must stay
independent metrics, not folded into one "leaning" number.
"""

import numpy as np

from dvi.geometry.normalize import to_body_frame
from dvi.schemas.frames import Frame
from dvi.schemas.metrics import MetricResult, TargetRange
from dvi.schemas.pose import PoseSequence


def _joint(pose: PoseSequence, name: str) -> np.ndarray:
    return pose.joints[:, pose.skeleton.joint_index(name)]


def _joints_valid(pose: PoseSequence, names: list[str]) -> np.ndarray:
    idxs = [pose.skeleton.joint_index(n) for n in names]
    return np.all(pose.valid[:, idxs], axis=-1)


def compute_pos01_series(pose: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    """Signed sagittal-plane torso pitch, +forward (ARCHITECTURE.md §6.1 POS-01).

    v = mid(hip_l, hip_r) -> neck; angle between v's sagittal-plane (Y-Z)
    projection and `up`, positive when leaning forward (+Z).
    """
    pose.require_frame(Frame.WORLD_G)
    hip_mid = (_joint(pose, "hip_l") + _joint(pose, "hip_r")) / 2
    v = _joint(pose, "neck") - hip_mid
    angle_deg = np.degrees(np.arctan2(v[:, 2], v[:, 1]))
    valid = _joints_valid(pose, ["hip_l", "hip_r", "neck"])
    return angle_deg, valid


def compute_pos03_series(pose: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    """Spine kink at spine_mid, as an unsigned deviation from 180°
    (ARCHITECTURE.md §6.1 POS-03) — 0 for a straight hip-spine_mid-neck line,
    growing with any fold regardless of direction.
    """
    pose.require_frame(Frame.WORLD_G)
    hip_mid = (_joint(pose, "hip_l") + _joint(pose, "hip_r")) / 2
    spine_mid = _joint(pose, "spine_mid")
    neck = _joint(pose, "neck")

    v1 = hip_mid - spine_mid
    v2 = neck - spine_mid
    cos_theta = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    theta_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    deviation_deg = np.abs(180.0 - theta_deg)

    valid = _joints_valid(pose, ["hip_l", "hip_r", "spine_mid", "neck"])
    return deviation_deg, valid


def compute_pos02_series(pose: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    """Signed coronal-plane torso roll (ARCHITECTURE.md §6.1 POS-02).

    Same torso vector as POS-01, projected onto the coronal (X-Y) plane
    instead of the sagittal one — independent axis, same construction.
    """
    pose.require_frame(Frame.WORLD_G)
    hip_mid = (_joint(pose, "hip_l") + _joint(pose, "hip_r")) / 2
    v = _joint(pose, "neck") - hip_mid
    angle_deg = np.degrees(np.arctan2(v[:, 0], v[:, 1]))
    valid = _joints_valid(pose, ["hip_l", "hip_r", "neck"])
    return angle_deg, valid


def compute_pos04_series(pose: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    """Forward head migration, BODY frame, normalized by torso length
    (ARCHITECTURE.md §6.1 POS-04) — signed, only forward drift is penalized
    per the salon.yaml target (max-only, no min).
    """
    pose.require_frame(Frame.WORLD_G)
    if pose.scale.torso_length_m is None:
        raise ValueError("POS-04 requires BodyScale.torso_length_m for normalization")
    body = to_body_frame(pose)
    forward_offset = (_joint(body, "head_top") - _joint(body, "neck"))[:, 2]
    series = forward_offset / pose.scale.torso_length_m
    valid = _joints_valid(body, ["head_top", "neck"])
    return series, valid


def compute_pos05a_series(pose: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    """Shoulder elevation ("shrug"), relative to this sequence's own low-percentile
    baseline (ARCHITECTURE.md §6.1 POS-05a).

    Pairwise Euclidean distance is rotation/translation-invariant, so
    WORLD_G vs. BODY frame makes no difference to the result here — computed
    directly on the input pose despite the architecture table's BODY tag.

    Deviates from the spec's "세션 초반 p10" (early-session p10) baseline in
    two ways: (1) uses the WHOLE provided sequence instead of an early
    window, since a session/warm-up boundary concept doesn't exist yet;
    (2) uses p90, not p10. Relaxed shoulders sit LOW, i.e. FAR from the
    ear (large distance); a shrug pulls the shoulder up, shrinking that
    distance. For `1 - dist/baseline` to read ~0 when relaxed and grow
    positive when shrugging, baseline must approximate the typical
    *relaxed* (large-distance) reading — a high percentile, not a low
    one. Taking the literal p10 would make baseline track the *most*
    shrugged moments instead, inverting the signal. Both deviations are
    candidates for revisiting once V4/V5 calibration data exists
    (PROJECT.md §7).
    """
    pose.require_frame(Frame.WORLD_G)
    dist_l = np.linalg.norm(_joint(pose, "shoulder_l") - _joint(pose, "ear_l"), axis=-1)
    dist_r = np.linalg.norm(_joint(pose, "shoulder_r") - _joint(pose, "ear_r"), axis=-1)
    dist = (dist_l + dist_r) / 2
    valid = _joints_valid(pose, ["shoulder_l", "ear_l", "shoulder_r", "ear_r"])

    if not valid.any():
        return np.full(dist.shape, np.nan), valid

    baseline = float(np.percentile(dist[valid], 90))
    series = 1.0 - dist / baseline
    return series, valid


def compute_pos05b_series(pose: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    """Shoulder line vs. horizontal (ARCHITECTURE.md §6.1 POS-05b) — signed
    angle, positive when the right shoulder sits higher than the left.
    """
    pose.require_frame(Frame.WORLD_G)
    v = _joint(pose, "shoulder_r") - _joint(pose, "shoulder_l")
    horizontal_len = np.linalg.norm(v[:, [0, 2]], axis=-1)
    angle_deg = np.degrees(np.arctan2(v[:, 1], horizontal_len))
    valid = _joints_valid(pose, ["shoulder_l", "shoulder_r"])
    return angle_deg, valid


def compute_pos06_series(pose: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    """Torso straightness (ARCHITECTURE.md §6.1 POS-06): RMS perpendicular
    distance of hip_mid/spine_mid/neck/head_top from their own best-fit 3D
    line, normalized by torso length. Vectorized across T via batched eigh
    on the per-frame 3x3 scatter matrix (ARCHITECTURE.md §10: no frame loops).
    """
    pose.require_frame(Frame.WORLD_G)
    if pose.scale.torso_length_m is None:
        raise ValueError("POS-06 requires BodyScale.torso_length_m for normalization")

    hip_mid = (_joint(pose, "hip_l") + _joint(pose, "hip_r")) / 2
    points = np.stack(
        [hip_mid, _joint(pose, "spine_mid"), _joint(pose, "neck"), _joint(pose, "head_top")], axis=1
    )  # (T, 4, 3)

    centered = points - points.mean(axis=1, keepdims=True)
    scatter = np.einsum("tik,til->tkl", centered, centered)  # (T, 3, 3)
    _, eigvecs = np.linalg.eigh(scatter)  # ascending eigenvalues
    direction = eigvecs[:, :, -1]  # (T, 3) principal axis (largest eigenvalue)

    proj = np.einsum("tij,tj->ti", centered, direction)  # (T, 4)
    perp = centered - proj[..., None] * direction[:, None, :]
    dist = np.linalg.norm(perp, axis=-1)  # (T, 4)
    residual = np.sqrt(np.mean(dist**2, axis=-1))  # (T,)

    series = residual / pose.scale.torso_length_m
    valid = _joints_valid(pose, ["hip_l", "hip_r", "spine_mid", "neck", "head_top"])
    return series, valid


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
        metric_id=metric_id,
        series=series,
        valid=valid,
        coverage=coverage,
        target=target,
        aggregate=aggregate,
        causal=causal,
    )


def pos01(pose: PoseSequence, target: TargetRange) -> MetricResult:
    series, valid = compute_pos01_series(pose)
    return _to_metric_result("POS-01", series, valid, target)


def pos03(pose: PoseSequence, target: TargetRange) -> MetricResult:
    series, valid = compute_pos03_series(pose)
    return _to_metric_result("POS-03", series, valid, target)


def pos02(pose: PoseSequence, target: TargetRange) -> MetricResult:
    series, valid = compute_pos02_series(pose)
    return _to_metric_result("POS-02", series, valid, target)


def pos04(pose: PoseSequence, target: TargetRange) -> MetricResult:
    series, valid = compute_pos04_series(pose)
    return _to_metric_result("POS-04", series, valid, target)


def pos05a(pose: PoseSequence, target: TargetRange) -> MetricResult:
    series, valid = compute_pos05a_series(pose)
    return _to_metric_result("POS-05a", series, valid, target)


def pos05b(pose: PoseSequence, target: TargetRange) -> MetricResult:
    series, valid = compute_pos05b_series(pose)
    return _to_metric_result("POS-05b", series, valid, target)


def pos06(pose: PoseSequence, target: TargetRange) -> MetricResult:
    series, valid = compute_pos06_series(pose)
    return _to_metric_result("POS-06", series, valid, target)
