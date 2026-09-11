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
