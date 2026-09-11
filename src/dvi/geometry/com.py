"""Center-of-mass estimation via segment mass weighting (ARCHITECTURE.md §6.2).

Uses representative Winter/Dempster-style anthropometric segment mass
fractions and proximal-COM ratios. This table is NOT finalized (open
question ARCHITECTURE.md §14 A2) — precision doesn't matter here, only
that the SAME table is applied consistently across a session, since AXI-*
metrics use COM for *relative* comparison, not absolute biomechanical
accuracy (§6.2: "정밀하진 않지만 일관적이면 충분하다").
"""

import numpy as np

from dvi.schemas.pose import PoseSequence

# (proximal_joint, distal_joint, mass_fraction, proximal_com_ratio)
# Mass fractions sum to ~1.0 across the full body. Hands are folded into
# the forearm segment and toes into the foot segment — DVI-Canonical has
# no separate joints for them.
_SEGMENTS: tuple[tuple[str, str, float, float], ...] = (
    ("neck", "head_top", 0.081, 0.50),
    ("pelvis", "neck", 0.497, 0.50),
    ("shoulder_l", "elbow_l", 0.028, 0.436),
    ("shoulder_r", "elbow_r", 0.028, 0.436),
    ("elbow_l", "wrist_l", 0.022, 0.682),
    ("elbow_r", "wrist_r", 0.022, 0.682),
    ("hip_l", "knee_l", 0.100, 0.433),
    ("hip_r", "knee_r", 0.100, 0.433),
    ("knee_l", "ankle_l", 0.0465, 0.433),
    ("knee_r", "ankle_r", 0.0465, 0.433),
    ("ankle_l", "foot_index_l", 0.0145, 0.50),
    ("ankle_r", "foot_index_r", 0.0145, 0.50),
)


def estimate_com(pose: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    """Weighted sum of per-segment COM positions. Returns (com, valid), each (T, 3)/(T,).

    A segment only contributes to a frame if both its endpoint joints are
    valid there; weights are renormalized over whatever segments *are*
    available that frame rather than requiring all of them. A frame with
    no valid segments at all gets NaN and valid=False — coarser than full
    per-joint confidence propagation, but keeps this self-contained
    (no FeatureSeries dependency).
    """
    t_len = pose.joints.shape[0]
    weighted_sum = np.zeros((t_len, 3))
    total_weight = np.zeros(t_len)

    for proximal, distal, mass_fraction, ratio in _SEGMENTS:
        if proximal not in pose.skeleton.joint_names or distal not in pose.skeleton.joint_names:
            continue
        p_idx = pose.skeleton.joint_index(proximal)
        d_idx = pose.skeleton.joint_index(distal)
        segment_valid = pose.valid[:, p_idx] & pose.valid[:, d_idx]

        p = pose.joints[:, p_idx]
        d = pose.joints[:, d_idx]
        segment_com = p + ratio * (d - p)

        weighted_sum[segment_valid] += mass_fraction * segment_com[segment_valid]
        total_weight[segment_valid] += mass_fraction

    valid = total_weight > 0
    com = np.full((t_len, 3), np.nan)
    com[valid] = weighted_sum[valid] / total_weight[valid, None]
    return com, valid
