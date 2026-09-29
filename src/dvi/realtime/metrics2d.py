"""Per-frame, causal 2D approximations of the POS-*/AXI-* metrics (ARCHITECTURE.md §11).

The offline metrics (metrics/posture.py, metrics/axis.py) are defined on a
gravity-aligned 3D PoseSequence (WORLD_G). The live path only has Tier A
2D keypoints from a single camera and ~40ms per frame, so there is no 3D
lift and no whole-sequence gravity estimate. Instead:

- Gravity = image vertical, corrected by a one-off roll calibration
  (§11 "세션 시작 시 1회 캘리브레이션 후 고정", reusing
  geometry/gravity.py's manual vertical-line fallback). A level camera
  needs no correction.
- Each metric is measured in the image plane, which only sees the body
  plane facing the camera. That's why metrics are tied to a camera `view`:
  a SIDE camera sees the sagittal plane (forward lean, head forward, knee
  angle), a FRONT camera sees the coronal plane (side lean, shoulder
  tilt). Measuring forward lean from a front camera is geometrically
  impossible in 2D, so those metrics are simply not reported there.

Metric IDs reuse the offline catalog IDs so the style profile targets and
coaching templates apply unchanged — but values are 2D projections, so
they read lower than the true 3D angle whenever the dancer isn't square
to the camera. Treat them as practice cues, not measurements (the same
caveat PROGRESS.md §4.3 makes for the synthetic-only offline metrics).
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class CameraView(StrEnum):
    SIDE = "side"
    FRONT = "front"


# Which metrics are observable in the image plane for each camera placement.
VIEW_METRICS: dict[CameraView, tuple[str, ...]] = {
    CameraView.SIDE: ("AXI-06", "POS-01", "POS-04", "POS-05a", "AXI-04"),
    CameraView.FRONT: ("AXI-06", "POS-02", "POS-05b", "POS-05a"),
}

METRIC_LABELS_KO: dict[str, str] = {
    "POS-01": "상체 전후 기울기",
    "POS-02": "상체 좌우 기울기",
    "POS-04": "머리 전방 이동",
    "POS-05a": "어깨 으쓱",
    "POS-05b": "어깨 수평",
    "AXI-04": "지지 무릎 각도",
    "AXI-06": "축 수직성",
}

# Offline catalog defines these as |value| targets (e.g. POS-02 "|θ| < 3°");
# the live path reports signed values for direction-specific cues but
# judges the magnitude.
ABSOLUTE_METRICS: frozenset[str] = frozenset({"POS-02", "POS-05b", "AXI-06"})

# Ankles closer than this (in torso lengths) count as double support — the
# axis is then measured from the ankle midpoint instead of one foot.
DOUBLE_SUPPORT_SPREAD = 0.25


@dataclass(frozen=True)
class Keypoints2D:
    """One frame's keypoints for one person, in `joint_names` order (image px, y-down)."""

    joint_names: tuple[str, ...]
    xy: np.ndarray  # (J, 2)
    valid: np.ndarray  # (J,) bool

    def has(self, *names: str) -> bool:
        return all(n in self.joint_names and self.valid[self.joint_names.index(n)] for n in names)

    def get(self, name: str) -> np.ndarray:
        return self.xy[self.joint_names.index(name)]


@dataclass(frozen=True)
class LiveCalibration:
    """Session-start calibration (ARCHITECTURE.md §11), fixed for the session.

    `roll_deg`: camera roll, from `manual_calibration_from_vertical_line`
    (0 for a level camera). `shrug_baseline`: relaxed shoulder-ear distance
    per torso length — POS-05a is relative to the dancer's own neutral,
    captured during the warm-up window.
    """

    roll_deg: float = 0.0
    shrug_baseline: float | None = None


def _to_gravity_up(p: np.ndarray, roll_deg: float) -> np.ndarray:
    """Image px (y-down) -> 2D gravity frame (x right, y up), undoing camera roll."""
    x, y = float(p[0]), -float(p[1])
    if roll_deg == 0.0:
        return np.array([x, y])
    r = np.radians(roll_deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([c * x - s * y, s * x + c * y])


def _angle_from_vertical_deg(v: np.ndarray) -> float:
    """Signed angle of v from +up, positive toward +x."""
    return float(np.degrees(np.arctan2(v[0], v[1])))


def _mid(kp: Keypoints2D, a: str, b: str) -> np.ndarray:
    return (kp.get(a) + kp.get(b)) / 2


def facing_sign(kp: Keypoints2D, roll_deg: float = 0.0) -> float | None:
    """+1 if the dancer faces image-right, -1 if image-left (side view only).

    The nose sits in front of the ears in profile, so nose-vs-ear x is a
    robust facing cue that needs no temporal state.
    """
    if not kp.has("nose"):
        return None
    ears = [kp.get(e) for e in ("ear_l", "ear_r") if kp.has(e)]
    if not ears:
        return None
    nose_x = _to_gravity_up(kp.get("nose"), roll_deg)[0]
    ear_x = np.mean([_to_gravity_up(e, roll_deg)[0] for e in ears])
    if abs(nose_x - ear_x) < 1e-6:
        return None
    return 1.0 if nose_x > ear_x else -1.0


def torso_length_px(kp: Keypoints2D) -> float | None:
    if not kp.has("hip_l", "hip_r", "shoulder_l", "shoulder_r"):
        return None
    length = float(np.linalg.norm(_mid(kp, "shoulder_l", "shoulder_r") - _mid(kp, "hip_l", "hip_r")))
    return length if length > 1e-6 else None


def support_side(kp: Keypoints2D) -> str | None:
    """'l', 'r', or 'both' — the leg the body is stacked over.

    Weight sits over the foot whose ankle is horizontally closest to the
    hip midpoint. A causal single-frame stand-in for the offline
    geometry/support_foot.py state machine, which needs 3D ankle heights.
    """
    if not kp.has("hip_l", "hip_r"):
        return None
    ankles = {s: kp.get(f"ankle_{s}") for s in ("l", "r") if kp.has(f"ankle_{s}")}
    if not ankles:
        return None
    if len(ankles) == 1:
        return next(iter(ankles))
    torso = torso_length_px(kp)
    spread = abs(ankles["l"][0] - ankles["r"][0])
    if torso is not None and spread / torso < DOUBLE_SUPPORT_SPREAD:
        return "both"
    hip_x = _mid(kp, "hip_l", "hip_r")[0]
    return min(ankles, key=lambda s: abs(ankles[s][0] - hip_x))


def pos01_2d(kp: Keypoints2D, cal: LiveCalibration) -> float | None:
    """Side view: signed torso pitch, +forward (hip_mid -> neck vs. up)."""
    if not kp.has("hip_l", "hip_r", "shoulder_l", "shoulder_r"):
        return None
    facing = facing_sign(kp, cal.roll_deg)
    if facing is None:
        return None
    hip = _to_gravity_up(_mid(kp, "hip_l", "hip_r"), cal.roll_deg)
    neck = _to_gravity_up(_mid(kp, "shoulder_l", "shoulder_r"), cal.roll_deg)
    return facing * _angle_from_vertical_deg(neck - hip)


def pos02_2d(kp: Keypoints2D, cal: LiveCalibration) -> float | None:
    """Front view: signed torso roll, + toward image-right."""
    if not kp.has("hip_l", "hip_r", "shoulder_l", "shoulder_r"):
        return None
    hip = _to_gravity_up(_mid(kp, "hip_l", "hip_r"), cal.roll_deg)
    neck = _to_gravity_up(_mid(kp, "shoulder_l", "shoulder_r"), cal.roll_deg)
    return _angle_from_vertical_deg(neck - hip)


def pos04_2d(kp: Keypoints2D, cal: LiveCalibration) -> float | None:
    """Side view: forward offset of the ears past the shoulders, per torso length.

    COCO-17 has no head_top, so the ear stands in for the head — it's also
    the classic craniovertebral landmark for forward head posture: stacked
    over the shoulder when aligned, drifting ahead of it when not.
    """
    facing = facing_sign(kp, cal.roll_deg)
    torso = torso_length_px(kp)
    ears = [kp.get(e) for e in ("ear_l", "ear_r") if kp.has(e)]
    if facing is None or torso is None or not ears:
        return None
    ear_x = np.mean([_to_gravity_up(e, cal.roll_deg)[0] for e in ears])
    neck_x = _to_gravity_up(_mid(kp, "shoulder_l", "shoulder_r"), cal.roll_deg)[0]
    return float(facing * (ear_x - neck_x) / torso)


def shoulder_ear_ratio(kp: Keypoints2D) -> float | None:
    """Mean shoulder-to-ear distance per torso length (POS-05a's raw quantity).

    Per-torso-length so walking toward/away from the camera doesn't read
    as a shrug.
    """
    torso = torso_length_px(kp)
    if torso is None:
        return None
    dists = [
        np.linalg.norm(kp.get(f"shoulder_{s}") - kp.get(f"ear_{s}"))
        for s in ("l", "r")
        if kp.has(f"shoulder_{s}", f"ear_{s}")
    ]
    if not dists:
        return None
    return float(np.mean(dists) / torso)


def pos05a_2d(kp: Keypoints2D, cal: LiveCalibration) -> float | None:
    """Shrug relative to this dancer's relaxed baseline: 1 - dist/baseline
    (same form as metrics/posture.py POS-05a — 0 relaxed, >0 shrugging)."""
    if cal.shrug_baseline is None:
        return None
    ratio = shoulder_ear_ratio(kp)
    if ratio is None:
        return None
    return 1.0 - ratio / cal.shrug_baseline


def pos05b_2d(kp: Keypoints2D, cal: LiveCalibration) -> float | None:
    """Front view: shoulder line vs. horizontal, + when the image-right shoulder is higher."""
    if not kp.has("shoulder_l", "shoulder_r"):
        return None
    a = _to_gravity_up(kp.get("shoulder_l"), cal.roll_deg)
    b = _to_gravity_up(kp.get("shoulder_r"), cal.roll_deg)
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    v = right - left
    return float(np.degrees(np.arctan2(v[1], abs(v[0]) + 1e-12)))


def axi04_2d(kp: Keypoints2D, cal: LiveCalibration) -> float | None:
    """Side view: support-leg knee angle, 180 = fully straight."""
    side = support_side(kp)
    if side is None:
        return None
    sides = ("l", "r") if side == "both" else (side,)
    angles = []
    for s in sides:
        if not kp.has(f"hip_{s}", f"knee_{s}", f"ankle_{s}"):
            continue
        knee = kp.get(f"knee_{s}")
        v1 = kp.get(f"hip_{s}") - knee
        v2 = kp.get(f"ankle_{s}") - knee
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom < 1e-9:
            continue
        angles.append(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))))
    return float(np.mean(angles)) if angles else None


def axi06_2d(kp: Keypoints2D, cal: LiveCalibration) -> float | None:
    """Axis verticality: support ankle -> body center vs. up.

    Body center is the pelvis-neck midpoint — COM of the trunk segment,
    which carries ~half the body mass (geometry/com.py's table); limbs are
    left out because Tier A's arm keypoints swing through the embrace.
    """
    side = support_side(kp)
    if side is None or not kp.has("shoulder_l", "shoulder_r"):
        return None
    if side == "both":
        if not kp.has("ankle_l", "ankle_r"):
            return None
        base = _mid(kp, "ankle_l", "ankle_r")
    else:
        base = kp.get(f"ankle_{side}")
    center = (_mid(kp, "hip_l", "hip_r") + _mid(kp, "shoulder_l", "shoulder_r")) / 2
    v = _to_gravity_up(center, cal.roll_deg) - _to_gravity_up(base, cal.roll_deg)
    return _angle_from_vertical_deg(v)


METRIC_FUNCTIONS = {
    "POS-01": pos01_2d,
    "POS-02": pos02_2d,
    "POS-04": pos04_2d,
    "POS-05a": pos05a_2d,
    "POS-05b": pos05b_2d,
    "AXI-04": axi04_2d,
    "AXI-06": axi06_2d,
}


def compute_live_metrics(
    kp: Keypoints2D, view: CameraView, cal: LiveCalibration
) -> dict[str, float | None]:
    """All metrics observable from `view`; None where required joints are missing."""
    return {mid: METRIC_FUNCTIONS[mid](kp, cal) for mid in VIEW_METRICS[view]}
