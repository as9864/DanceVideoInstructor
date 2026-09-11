import numpy as np
import pytest

from dvi.schemas.metrics import MetricResult, TargetRange
from eval.harness import angle_consistency

TARGET = TargetRange(unit="deg", max=5.0)


def make_result(metric_id: str, series: np.ndarray, valid: np.ndarray | None = None) -> MetricResult:
    valid = np.ones_like(series, dtype=bool) if valid is None else valid
    return MetricResult(
        metric_id=metric_id,
        series=series,
        valid=valid,
        coverage=float(valid.mean()),
        target=TARGET,
        aggregate={"mean": float(series[valid].mean()) if valid.any() else float("nan")},
        causal=True,
    )


class TestAngleConsistency:
    def test_identical_angles_have_zero_std(self):
        series = np.array([1.0, 2.0, 3.0, 4.0])
        results = [make_result("POS-01", series.copy()) for _ in range(3)]
        result = angle_consistency(results)

        assert result.mean_std == pytest.approx(0.0, abs=1e-12)
        assert result.max_std == pytest.approx(0.0, abs=1e-12)
        assert result.n_frames_compared == 4
        assert result.coverage == 1.0

    def test_diverging_angles_produce_positive_std(self):
        results = [
            make_result("POS-01", np.array([0.0, 0.0, 0.0])),
            make_result("POS-01", np.array([10.0, 10.0, 10.0])),
        ]
        result = angle_consistency(results)
        assert result.mean_std == pytest.approx(5.0)  # std of {0, 10} is 5

    def test_frame_invalid_in_one_angle_excluded_from_comparison(self):
        valid_a = np.array([True, True, True])
        valid_b = np.array([True, False, True])  # frame 1 occluded in angle B
        results = [
            make_result("POS-01", np.array([1.0, 999.0, 1.0]), valid_a),
            make_result("POS-01", np.array([1.0, -999.0, 1.0]), valid_b),
        ]
        result = angle_consistency(results)
        # only frames 0 and 2 are valid in both angles, and both agree exactly there
        assert result.n_frames_compared == 2
        assert result.mean_std == pytest.approx(0.0, abs=1e-12)
        assert result.coverage == pytest.approx(2 / 3)

    def test_no_frame_valid_everywhere_gives_nan_and_zero_coverage(self):
        results = [
            make_result("POS-01", np.array([1.0, 2.0]), np.array([True, False])),
            make_result("POS-01", np.array([1.0, 2.0]), np.array([False, True])),
        ]
        result = angle_consistency(results)
        assert result.n_frames_compared == 0
        assert result.coverage == 0.0
        assert np.isnan(result.mean_std)

    def test_requires_at_least_two_angles(self):
        with pytest.raises(ValueError, match="at least 2 angles"):
            angle_consistency([make_result("POS-01", np.array([1.0]))])

    def test_mismatched_metric_id_raises(self):
        with pytest.raises(ValueError, match="metric_id"):
            angle_consistency(
                [make_result("POS-01", np.array([1.0, 2.0])), make_result("POS-03", np.array([1.0, 2.0]))]
            )

    def test_mismatched_series_length_raises(self):
        with pytest.raises(ValueError, match="synchronized"):
            angle_consistency(
                [make_result("POS-01", np.array([1.0, 2.0])), make_result("POS-01", np.array([1.0, 2.0, 3.0]))]
            )
