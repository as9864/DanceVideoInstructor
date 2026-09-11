"""Synthetic/geometric validation of the gravity fallback chain (PROJECT.md §7 V1).

Each estimator only fixes pitch/roll (WORLD_G leaves yaw unconstrained,
ARCHITECTURE.md §4.1), so tests check invariants — "the floor is flat after
rotation", "the body ends up above it" — rather than recovering the exact
original rotation matrix.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from dvi.geometry.calibration import CalibrationProfile
from dvi.geometry.gravity import (
    estimate_from_floor_plane,
    estimate_from_long_axis_prior,
    estimate_gravity,
    manual_calibration_from_vertical_line,
)

RNG = np.random.default_rng(0)


def _tilted_floor_and_body(tilt_xyz_deg, n_floor=200, n_body=50, noise=0.002, n_outliers=0):
    """Floor points on the true y=0 plane and body points at y>0, both
    transformed into a "pre-alignment" frame by a known tilt rotation."""
    r_true = Rotation.from_euler("xyz", tilt_xyz_deg, degrees=True).as_matrix()

    world_floor = np.stack(
        [RNG.uniform(-1, 1, n_floor), np.zeros(n_floor), RNG.uniform(-1, 1, n_floor)], axis=1
    )
    world_floor += RNG.normal(scale=noise, size=world_floor.shape)
    if n_outliers:
        outliers = RNG.uniform(-1, 1, size=(n_outliers, 3))
        world_floor = np.concatenate([world_floor, outliers], axis=0)

    world_body = np.stack(
        [RNG.uniform(-0.3, 0.3, n_body), RNG.uniform(0.5, 1.8, n_body), RNG.uniform(-0.3, 0.3, n_body)],
        axis=1,
    )

    camera_floor = world_floor @ r_true
    camera_body = world_body @ r_true
    return camera_floor, camera_body


class TestFloorPlaneRANSAC:
    def test_recovers_flat_floor_and_body_above_it(self):
        camera_floor, camera_body = _tilted_floor_and_body([8.0, -5.0, 0.0])
        profile = estimate_from_floor_plane(camera_floor, camera_body)

        assert profile is not None
        assert profile.method == "floor_plane_ransac"
        assert profile.confidence > 0.8

        recovered_floor = profile.apply(camera_floor)
        recovered_body = profile.apply(camera_body)

        assert recovered_floor[:, 1].std() < 0.01  # floor is flat in Y after alignment
        assert recovered_body[:, 1].min() > recovered_floor[:, 1].max()  # body strictly above floor

    def test_severe_tilt_still_recovered(self):
        camera_floor, camera_body = _tilted_floor_and_body([25.0, 15.0, 40.0])
        profile = estimate_from_floor_plane(camera_floor, camera_body)
        assert profile is not None
        recovered_floor = profile.apply(camera_floor)
        assert recovered_floor[:, 1].std() < 0.01

    def test_too_few_points_returns_none(self):
        assert estimate_from_floor_plane(np.zeros((2, 3)), np.zeros((5, 3))) is None

    def test_mostly_outliers_returns_none(self):
        camera_floor, camera_body = _tilted_floor_and_body([8.0, 0.0, 0.0], n_floor=20, n_outliers=200)
        assert estimate_from_floor_plane(camera_floor, camera_body, min_inliers_fraction=0.5) is None


class TestLongAxisPrior:
    def test_recovers_mean_direction_as_up(self):
        r_true = Rotation.from_euler("xyz", [6.0, 3.0, 0.0], degrees=True).as_matrix()
        world_torso = np.tile([0.0, 1.0, 0.0], (100, 1)) + RNG.normal(scale=0.01, size=(100, 3))
        camera_torso = world_torso @ r_true

        profile = estimate_from_long_axis_prior(camera_torso)
        recovered = profile.apply(camera_torso)
        mean_recovered = recovered.mean(axis=0)
        mean_recovered /= np.linalg.norm(mean_recovered)

        assert mean_recovered == pytest.approx([0.0, 1.0, 0.0], abs=1e-2)
        assert profile.confidence > 0.99  # low noise -> tight cluster around the mean

    def test_noisy_directions_get_lower_confidence(self):
        scattered = RNG.normal(size=(200, 3))  # no consistent direction at all
        profile = estimate_from_long_axis_prior(scattered)
        assert profile.confidence < 0.5

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            estimate_from_long_axis_prior(np.zeros((0, 3)))


class TestManualCalibration:
    def test_already_vertical_line_gives_identity_rotation(self):
        profile = manual_calibration_from_vertical_line((10.0, 50.0), (10.0, 0.0))
        assert profile.metadata["roll_deg"] == pytest.approx(0.0, abs=1e-6)
        assert np.allclose(profile.rotation, np.eye(3), atol=1e-6)

    def test_tilted_line_recovers_known_angle(self):
        theta_deg = 12.0
        theta = np.radians(theta_deg)
        length = 100.0
        point_a = (0.0, 0.0)
        point_b = (length * np.sin(theta), -length * np.cos(theta))  # tilted `theta` from vertical
        profile = manual_calibration_from_vertical_line(point_a, point_b)
        assert profile.metadata["roll_deg"] == pytest.approx(theta_deg, abs=1e-6)
        assert profile.confidence == 1.0

    def test_identical_points_raise(self):
        with pytest.raises(ValueError):
            manual_calibration_from_vertical_line((5.0, 5.0), (5.0, 5.0))


class TestEstimateGravityFallbackChain:
    def test_prefers_floor_plane_when_it_succeeds(self):
        camera_floor, camera_body = _tilted_floor_and_body([8.0, 0.0, 0.0])
        torso_vectors = np.tile([0.0, 1.0, 0.5], (50, 1))  # deliberately different, should be ignored

        profile = estimate_gravity(
            grounded_points=camera_floor, body_points_for_sign=camera_body, torso_vectors=torso_vectors
        )
        assert profile.method == "floor_plane_ransac"

    def test_falls_back_to_long_axis_when_floor_fit_fails(self):
        torso_vectors = np.tile([0.1, 1.0, 0.0], (50, 1))
        profile = estimate_gravity(
            grounded_points=np.zeros((1, 3)),  # too few points -> floor fit fails
            body_points_for_sign=np.zeros((5, 3)),
            torso_vectors=torso_vectors,
        )
        assert profile.method == "long_axis_prior"

    def test_falls_back_to_manual_when_nothing_else_available(self):
        profile = estimate_gravity(manual_points=((0.0, 0.0), (0.0, -10.0)))
        assert profile.method == "manual"

    def test_raises_when_no_input_provided(self):
        with pytest.raises(ValueError):
            estimate_gravity()


class TestCalibrationProfile:
    def test_rejects_wrong_rotation_shape(self):
        with pytest.raises(ValueError):
            CalibrationProfile(method="manual", rotation=np.eye(2), confidence=1.0)

    def test_rejects_out_of_range_confidence(self):
        with pytest.raises(ValueError):
            CalibrationProfile(method="manual", rotation=np.eye(3), confidence=1.5)

    def test_apply_only_rotates_no_translation(self):
        profile = CalibrationProfile(method="manual", rotation=np.eye(3), confidence=1.0)
        points = np.array([[1.0, 2.0, 3.0]])
        assert np.allclose(profile.apply(points), points)
