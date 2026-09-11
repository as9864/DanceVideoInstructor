"""Synthetic canonical skeletons with exact, known posture angles.

Built for PROJECT.md §7 V1 ("합성/기하 검증"): feed a skeleton with a known
10° tilt, assert the metric reports 10°. Only the quantities a given metric
actually reads (pelvis/neck/spine_mid geometry) are geometrically exact —
everything else (arms, legs, head) is a plausible placeholder so the full
21-joint PoseSequence schema is satisfied.

Math notes:

- Torso direction is built directly from its own sagittal/coronal angles
  rather than composing two axis rotations, so POS-01 (sagittal) and
  POS-02 (coronal) can be set independently:
      v = (cos(pitch)*tan(roll), cos(pitch), sin(pitch)), normalized
  gives atan2(v.z, v.y) == pitch and atan2(v.x, v.y) == roll exactly,
  because both projections divide by the same v.y and the tan/atan2 cancel.

- spine_mid is placed off the pelvis-neck line as the apex of an isoceles
  triangle with base = |neck - pelvis| and a chosen apex angle, so the
  angle POS-03 measures at spine_mid comes out to exactly the requested
  180-kyphosis_deg regardless of pelvis/neck's orientation in space
  (the construction only depends on Euclidean distances, which are
  rotation-invariant) — see `_spine_mid_offset`.
"""

import numpy as np

from dvi.schemas import BodyScale, Frame, PoseSequence, Provenance
from dvi.schemas.skeleton import DVI_CANONICAL_V1

_FORWARD = np.array([0.0, 0.0, 1.0])
_UP = np.array([0.0, 1.0, 0.0])
_SIDE = np.array([1.0, 0.0, 0.0])


def _torso_direction(pitch_deg: float, roll_deg: float) -> np.ndarray:
    pitch = np.radians(pitch_deg)
    roll = np.radians(roll_deg)
    y = np.cos(pitch)
    z = np.sin(pitch)
    x = y * np.tan(roll)
    v = np.array([x, y, z])
    return v / np.linalg.norm(v)


def _spine_mid_offset(pelvis: np.ndarray, neck: np.ndarray, kyphosis_deg: float) -> np.ndarray:
    straight_mid = (pelvis + neck) / 2
    kyphosis_deg = abs(kyphosis_deg)
    if kyphosis_deg == 0.0:
        return straight_mid

    d = neck - pelvis
    dist = np.linalg.norm(d)
    u = d / dist

    perp = _FORWARD - np.dot(_FORWARD, u) * u
    if np.linalg.norm(perp) < 1e-9:
        perp = _UP - np.dot(_UP, u) * u  # torso ~parallel to FORWARD (extreme pitch): bend sideways instead
    perp = perp / np.linalg.norm(perp)

    theta_deg = 180.0 - kyphosis_deg
    half_base = dist / 2
    h = half_base / np.tan(np.radians(theta_deg) / 2)
    return straight_mid + h * perp


def build_synthetic_standing_pose(
    torso_pitch_deg: float = 0.0,
    torso_roll_deg: float = 0.0,
    spine_kyphosis_deg: float = 0.0,
    n_frames: int = 1,
    fps: float = 30.0,
    torso_length_m: float = 0.5,
    pelvis_height_m: float = 0.9,
    hip_width_m: float = 0.3,
    shoulder_width_m: float = 0.4,
    subject_id: str = "synthetic",
) -> PoseSequence:
    """A static (or repeated) standing pose with exact torso-tilt/kyphosis angles.

    frame=WORLD_G, every joint valid with confidence 1.0 — the goal is to
    isolate the metric's geometry, not to exercise the confidence/coverage
    machinery (that's a separate test concern).
    """
    pelvis = np.array([0.0, pelvis_height_m, 0.0])
    torso_dir = _torso_direction(torso_pitch_deg, torso_roll_deg)
    neck = pelvis + torso_length_m * torso_dir
    spine_mid = _spine_mid_offset(pelvis, neck, spine_kyphosis_deg)
    head_top = neck + 0.25 * torso_dir
    nose = head_top + np.array([0.0, -0.05, 0.08])
    ear_l = head_top - _SIDE * 0.08 + np.array([0.0, -0.05, 0.0])
    ear_r = head_top + _SIDE * 0.08 + np.array([0.0, -0.05, 0.0])

    hip_l = pelvis - _SIDE * hip_width_m / 2
    hip_r = pelvis + _SIDE * hip_width_m / 2
    shoulder_l = neck - _SIDE * shoulder_width_m / 2
    shoulder_r = neck + _SIDE * shoulder_width_m / 2
    elbow_l = shoulder_l + np.array([0.0, -0.25, 0.0])
    elbow_r = shoulder_r + np.array([0.0, -0.25, 0.0])
    wrist_l = elbow_l + np.array([0.0, -0.25, 0.0])
    wrist_r = elbow_r + np.array([0.0, -0.25, 0.0])
    knee_l = hip_l + np.array([0.0, -0.45, 0.0])
    knee_r = hip_r + np.array([0.0, -0.45, 0.0])
    ankle_l = knee_l + np.array([0.0, -0.45, 0.0])
    ankle_r = knee_r + np.array([0.0, -0.45, 0.0])
    foot_index_l = ankle_l + np.array([0.0, -0.05, 0.10])
    foot_index_r = ankle_r + np.array([0.0, -0.05, 0.10])
    heel_l = ankle_l + np.array([0.0, -0.05, -0.05])
    heel_r = ankle_r + np.array([0.0, -0.05, -0.05])

    joint_positions = {
        "pelvis": pelvis,
        "spine_mid": spine_mid,
        "neck": neck,
        "head_top": head_top,
        "nose": nose,
        "ear_l": ear_l,
        "ear_r": ear_r,
        "shoulder_l": shoulder_l,
        "shoulder_r": shoulder_r,
        "elbow_l": elbow_l,
        "elbow_r": elbow_r,
        "wrist_l": wrist_l,
        "wrist_r": wrist_r,
        "hip_l": hip_l,
        "hip_r": hip_r,
        "knee_l": knee_l,
        "knee_r": knee_r,
        "ankle_l": ankle_l,
        "ankle_r": ankle_r,
        "foot_index_l": foot_index_l,
        "foot_index_r": foot_index_r,
        "heel_l": heel_l,
        "heel_r": heel_r,
    }

    skeleton = DVI_CANONICAL_V1
    single_frame = np.stack([joint_positions[name] for name in skeleton.joint_names])  # (J, 3)
    joints = np.tile(single_frame, (n_frames, 1, 1))  # (T, J, 3)
    confidence = np.ones((n_frames, skeleton.n_joints))
    valid = np.ones((n_frames, skeleton.n_joints), dtype=bool)
    t = np.arange(n_frames, dtype=float) / fps

    return PoseSequence(
        subject_id=subject_id,
        frame=Frame.WORLD_G,
        fps=fps,
        t=t,
        joints=joints,
        confidence=confidence,
        valid=valid,
        skeleton=skeleton,
        scale=BodyScale(torso_length_m=torso_length_m),
        provenance=Provenance(backend_name="synthetic", backend_version="0", config_hash="synthetic"),
    )
