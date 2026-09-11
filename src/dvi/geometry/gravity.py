"""GravityEstimator fallback chain (ARCHITECTURE.md §5.2, PROJECT.md R2/R3).

"상체가 얼마나 쓰러졌나" only means anything relative to gravity, not to
whatever the camera happened to be pointed at. This module turns 3D points
in whatever orientation the pre-alignment stage produced them in into
WORLD_G (+Y up), using the priority chain from ARCHITECTURE.md §5.2:

  1. model-provided   — a Tier C backend with provides_world_frame=True
                         already IS WORLD_G; this module has nothing to do
                         and is skipped entirely by the caller.
  2. floor_plane_ransac — fit a plane to points believed to touch the floor.
  3. long_axis_prior    — assume the time-averaged torso direction is ~vertical.
  4. manual             — user marks a known-vertical line in one frame.

Each function is deliberately convention-agnostic about which axis "up"
starts out as — a floor-plane fit only produces a normal direction, and the
sign of that normal (which side the body is on) is resolved by checking
which side of the plane the body actually sits on, not by assuming a
specific camera axis convention. Reconciling this with a specific Lift3D
backend's actual output convention is integration work for when a Tier B/C
backend is chosen (ADR-0001 is still Proposed).
"""

import numpy as np
from scipy.spatial.transform import Rotation

from dvi.geometry.calibration import CalibrationProfile

_WORLD_UP = np.array([0.0, 1.0, 0.0])


def _rotation_aligning_to_up(v: np.ndarray) -> np.ndarray:
    """The minimal rotation mapping unit vector v onto WORLD_G's +Y.

    Deliberately does not touch the component of the frame orthogonal to
    the tilt plane (no arbitrary yaw is introduced) — WORLD_G only pins
    down "up" (ARCHITECTURE.md §4.1), so the smallest rotation that
    achieves that is the reproducible, deterministic choice (NFR-03).
    """
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("cannot align a near-zero vector")
    v = v / norm
    rot, _ = Rotation.align_vectors([_WORLD_UP], [v])
    return rot.as_matrix()


def estimate_from_floor_plane(
    grounded_points: np.ndarray,
    body_points_for_sign: np.ndarray,
    n_iterations: int = 200,
    inlier_threshold: float = 0.02,
    min_inliers_fraction: float = 0.5,
    rng: np.random.Generator | None = None,
) -> CalibrationProfile | None:
    """ARCHITECTURE.md §5.2 priority 2.

    `grounded_points` (N, 3): points believed to be in contact with the
    floor (e.g. from a future ground-contact state machine, AXI-05).
    `body_points_for_sign` (M, 3): any points known to be above the floor
    (e.g. all joints across the sequence) — used only to pick which of the
    two normal directions points "up", never to fit the plane itself.

    Returns None if there isn't enough grounded-point support for a stable
    fit (PROJECT.md R1: dancing in place gives too few ground-contact
    samples) — callers should fall back to `estimate_from_long_axis_prior`.
    """
    n = grounded_points.shape[0]
    if n < 3:
        return None
    rng = rng or np.random.default_rng()

    best_inlier_count = 0
    best_sample_point: np.ndarray | None = None
    best_normal: np.ndarray | None = None

    for _ in range(n_iterations):
        i0, i1, i2 = rng.choice(n, size=3, replace=False)
        p0, p1, p2 = grounded_points[i0], grounded_points[i1], grounded_points[i2]
        normal = np.cross(p1 - p0, p2 - p0)
        normal_len = np.linalg.norm(normal)
        if normal_len < 1e-9:
            continue
        normal = normal / normal_len
        distances = np.abs((grounded_points - p0) @ normal)
        inlier_count = int(np.sum(distances < inlier_threshold))
        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_sample_point = p0

    if best_normal is None or best_inlier_count / n < min_inliers_fraction:
        return None

    # Refine with a least-squares fit (SVD) over the inlier set found above.
    distances = np.abs((grounded_points - best_sample_point) @ best_normal)
    inlier_mask = distances < inlier_threshold
    inliers = grounded_points[inlier_mask]
    centroid = inliers.mean(axis=0)
    _, _, vt = np.linalg.svd(inliers - centroid)
    normal = vt[-1]
    normal = normal / np.linalg.norm(normal)

    body_centroid = body_points_for_sign.mean(axis=0)
    if np.dot(body_centroid - centroid, normal) < 0:
        normal = -normal

    inlier_fraction = float(inlier_mask.mean())
    return CalibrationProfile(
        method="floor_plane_ransac",
        rotation=_rotation_aligning_to_up(normal),
        confidence=inlier_fraction,
        metadata={"inlier_fraction": inlier_fraction, "n_points": float(n)},
    )


def estimate_from_long_axis_prior(torso_vectors: np.ndarray) -> CalibrationProfile:
    """ARCHITECTURE.md §5.2 priority 3.

    `torso_vectors` (T, 3): pelvis->neck vectors over a long window. Always
    "succeeds" (it's a prior, not a fit with a failure mode) — confidence
    reflects how tightly torso direction actually clustered around its own
    mean, which is a proxy for how safe the "mostly vertical" assumption
    was, not for absolute correctness (PROJECT.md R4: no ground truth here).
    """
    if torso_vectors.shape[0] == 0:
        raise ValueError("need at least one torso vector")

    norms = np.linalg.norm(torso_vectors, axis=1, keepdims=True)
    unit_vectors = torso_vectors / norms
    mean_dir = unit_vectors.mean(axis=0)
    mean_dir = mean_dir / np.linalg.norm(mean_dir)

    cos_similarity = unit_vectors @ mean_dir
    confidence = float(np.clip(cos_similarity.mean(), 0.0, 1.0))

    return CalibrationProfile(
        method="long_axis_prior",
        rotation=_rotation_aligning_to_up(mean_dir),
        confidence=confidence,
        metadata={"mean_cos_similarity": confidence, "n_frames": float(torso_vectors.shape[0])},
    )


def manual_calibration_from_vertical_line(
    point_a_px: tuple[float, float], point_b_px: tuple[float, float]
) -> CalibrationProfile:
    """ARCHITECTURE.md §5.2 priority 4.

    The user marks two pixel points on a known-vertical real-world line
    (wall edge, doorframe, pillar). A single 2D line only constrains the
    camera's in-plane roll — correcting perspective pitch from one 2D line
    would need focal length/intrinsics we don't have at this stage, so
    this fallback is intentionally roll-only, not a full correction.
    """
    ax, ay = point_a_px
    bx, by = point_b_px
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        raise ValueError("the two calibration points must be distinct")

    # Image y grows downward; angle of the line from image-vertical.
    roll_rad = np.arctan2(dx, -dy)
    rotation = Rotation.from_euler("z", -roll_rad).as_matrix()

    return CalibrationProfile(
        method="manual",
        rotation=rotation,
        confidence=1.0,  # user-provided; trusted at face value
        metadata={"roll_deg": float(np.degrees(roll_rad))},
    )


def estimate_gravity(
    *,
    grounded_points: np.ndarray | None = None,
    body_points_for_sign: np.ndarray | None = None,
    torso_vectors: np.ndarray | None = None,
    manual_points: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> CalibrationProfile:
    """Runs the ARCHITECTURE.md §5.2 fallback chain, priorities 2-4.

    Priority 1 (model-provided) isn't handled here — a Tier C backend with
    provides_world_frame=True means GravityAlign has nothing to do at all,
    so the caller should skip this function entirely in that case.
    """
    if grounded_points is not None and body_points_for_sign is not None:
        result = estimate_from_floor_plane(grounded_points, body_points_for_sign)
        if result is not None:
            return result

    if torso_vectors is not None:
        return estimate_from_long_axis_prior(torso_vectors)

    if manual_points is not None:
        return manual_calibration_from_vertical_line(*manual_points)

    raise ValueError(
        "no gravity estimation method had sufficient input "
        "(floor plane fit failed or wasn't attempted, no torso_vectors, no manual_points)"
    )
