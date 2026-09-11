"""Backend joint-set adapters -> DVI-Canonical (ARCHITECTURE.md §4.3, ADR-0001 §3).

ADR-0001's M1 scaffold candidate (usyd-community/vitpose-plus-base) outputs
COCO-17, which has neither pelvis/neck/spine nor any foot keypoint. pelvis/
neck/spine_mid are safe to derive (midpoints of joints COCO-17 does have).
Foot keypoints are NOT derivable — ADR-0001 §2 C1 is explicit that heel/toe
positions cannot be inferred from the ankle — so those stay invalid until
the backend is swapped for a Halpe-26 model before M3.
"""

import numpy as np

from dvi.schemas.skeleton import SkeletonSpec

COCO17_JOINT_NAMES: tuple[str, ...] = (
    "nose",
    "eye_l",
    "eye_r",
    "ear_l",
    "ear_r",
    "shoulder_l",
    "shoulder_r",
    "elbow_l",
    "elbow_r",
    "wrist_l",
    "wrist_r",
    "hip_l",
    "hip_r",
    "knee_l",
    "knee_r",
    "ankle_l",
    "ankle_r",
)

# raw COCO-17 name -> DVI-Canonical name, for joints with a direct 1:1 match
_COCO17_DIRECT: dict[str, str] = {
    "nose": "nose",
    "ear_l": "ear_l",
    "ear_r": "ear_r",
    "shoulder_l": "shoulder_l",
    "shoulder_r": "shoulder_r",
    "elbow_l": "elbow_l",
    "elbow_r": "elbow_r",
    "wrist_l": "wrist_l",
    "wrist_r": "wrist_r",
    "hip_l": "hip_l",
    "hip_r": "hip_r",
    "knee_l": "knee_l",
    "knee_r": "knee_r",
    "ankle_l": "ankle_l",
    "ankle_r": "ankle_r",
}

# The canonical joints a COCO-17 backend can populate at all: direct matches
# plus pelvis/neck/spine_mid derived as midpoints. head_top and every foot
# joint are excluded entirely rather than faked (no reliable derivation).
COCO17_SKELETON_SPEC = SkeletonSpec(
    name="dvi-canonical-from-coco17",
    joint_names=(
        "pelvis",
        "spine_mid",
        "neck",
        "nose",
        "ear_l",
        "ear_r",
        "shoulder_l",
        "shoulder_r",
        "elbow_l",
        "elbow_r",
        "wrist_l",
        "wrist_r",
        "hip_l",
        "hip_r",
        "knee_l",
        "knee_r",
        "ankle_l",
        "ankle_r",
    ),
    derived_joints=frozenset({"pelvis", "spine_mid", "neck"}),
)


def map_coco17_to_canonical(
    keypoints: np.ndarray,
    scores: np.ndarray,
    confidence_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map one frame's (17, 2) COCO-17 keypoints + (17,) scores to
    (J, 2)/(J,)/(J,) arrays ordered per COCO17_SKELETON_SPEC.

    Derived joints get `min()` confidence of their inputs — a derived point
    is only as trustworthy as its shakiest source keypoint.
    """
    if keypoints.shape != (17, 2):
        raise ValueError(f"expected (17, 2) COCO-17 keypoints, got {keypoints.shape}")
    if scores.shape != (17,):
        raise ValueError(f"expected (17,) COCO-17 scores, got {scores.shape}")

    spec = COCO17_SKELETON_SPEC
    j = spec.n_joints
    out_kpts = np.zeros((j, 2), dtype=float)
    out_conf = np.zeros(j, dtype=float)

    raw_index = {name: i for i, name in enumerate(COCO17_JOINT_NAMES)}

    for raw_name, canon_name in _COCO17_DIRECT.items():
        i = raw_index[raw_name]
        out_kpts[spec.joint_index(canon_name)] = keypoints[i]
        out_conf[spec.joint_index(canon_name)] = scores[i]

    def canon(name: str) -> tuple[np.ndarray, float]:
        idx = spec.joint_index(name)
        return out_kpts[idx], out_conf[idx]

    hip_l, c_hl = canon("hip_l")
    hip_r, c_hr = canon("hip_r")
    pelvis_idx = spec.joint_index("pelvis")
    out_kpts[pelvis_idx] = (hip_l + hip_r) / 2
    out_conf[pelvis_idx] = min(c_hl, c_hr)

    sh_l, c_sl = canon("shoulder_l")
    sh_r, c_sr = canon("shoulder_r")
    neck_idx = spec.joint_index("neck")
    out_kpts[neck_idx] = (sh_l + sh_r) / 2
    out_conf[neck_idx] = min(c_sl, c_sr)

    pelvis, c_p = canon("pelvis")
    neck, c_n = canon("neck")
    spine_idx = spec.joint_index("spine_mid")
    out_kpts[spine_idx] = (pelvis + neck) / 2
    out_conf[spine_idx] = min(c_p, c_n)

    out_valid = out_conf >= confidence_threshold
    return out_kpts, out_conf, out_valid
