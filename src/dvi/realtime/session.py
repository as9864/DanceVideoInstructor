"""LiveSession — one frame in, keypoints + metrics + feedback out (ARCHITECTURE.md §11).

Everything here is causal and per-frame; the camera loop, display and
sound live in pipeline/stream_runner.py so this stays testable with a
fake pose estimator and synthetic images (design principle P3).
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from dvi.geometry.smoothing import OneEuroFilter
from dvi.realtime.feedback import FeedbackEngine, FeedbackUpdate
from dvi.realtime.metrics2d import (
    CameraView,
    Keypoints2D,
    LiveCalibration,
    compute_live_metrics,
    shoulder_ear_ratio,
)
from dvi.schemas.metrics import TargetRange

BoxXYWH = tuple[float, float, float, float]


class LivePoseEstimator(Protocol):
    """Single-frame, single-person 2D pose. `box` None = whole frame."""

    joint_names: tuple[str, ...]

    def estimate(self, image_rgb: np.ndarray, box: BoxXYWH | None) -> tuple[np.ndarray, np.ndarray]:
        """Returns ((J, 2) keypoints in image px, (J,) confidence)."""
        ...


@dataclass
class LiveFrameResult:
    t: float
    keypoints: Keypoints2D | None  # smoothed; None when nobody is found
    box: BoxXYWH | None
    values: dict[str, float | None]
    feedback: FeedbackUpdate | None  # None while calibrating
    calibration_progress: float | None  # 0..1 while calibrating, else None


def box_from_keypoints(
    xy: np.ndarray, valid: np.ndarray, image_size: tuple[int, int], margin: float = 0.25, min_points: int = 6
) -> BoxXYWH | None:
    """Next frame's person box from this frame's keypoints (top-down tracking).

    Tier A runs one box per frame; a whole-frame box wastes most of the
    model's input resolution on background when the dancer stands far from
    the camera. Reusing the last pose's extent keeps the person large in
    the crop, at zero extra cost — no detector in the loop. Falls back to
    the whole frame (None) when too few joints were found to trust it.
    """
    if int(valid.sum()) < min_points:
        return None
    pts = xy[valid]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bw, bh = x1 - x0, y1 - y0
    # extra headroom above: COCO-17 tops out at the eyes/ears
    x0, x1 = x0 - bw * margin, x1 + bw * margin
    y0, y1 = y0 - bh * margin * 1.2, y1 + bh * margin * 0.6
    w, h = image_size
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(w), x1), min(float(h), y1)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


class LiveSession:
    def __init__(
        self,
        estimator: LivePoseEstimator,
        targets: dict[str, TargetRange],
        messages: dict[str, dict[str, str]],
        *,
        view: CameraView = CameraView.SIDE,
        roll_deg: float = 0.0,
        calibration_s: float = 3.0,
        min_calibration_samples: int = 10,
        confidence_threshold: float = 0.3,
        min_valid_joints: int = 8,
        mirrored: bool = False,
        hold_s: float = 0.3,
        release_s: float = 0.5,
        cooldown_s: float = 4.0,
        smoothing_min_cutoff: float = 1.0,
        smoothing_beta: float = 5.0,
    ):
        self.estimator = estimator
        self.targets = targets
        self.view = view
        self.roll_deg = roll_deg
        self.calibration_s = calibration_s
        self.min_calibration_samples = min_calibration_samples
        self.confidence_threshold = confidence_threshold
        self.min_valid_joints = min_valid_joints
        # Filtering in image-height units keeps min_cutoff/beta resolution-independent.
        self.filter = OneEuroFilter(min_cutoff=smoothing_min_cutoff, beta=smoothing_beta)
        self.feedback = FeedbackEngine(
            targets, messages, hold_s=hold_s, release_s=release_s, cooldown_s=cooldown_s, mirrored=mirrored
        )
        self.calibration = LiveCalibration(roll_deg=roll_deg)
        self._box: BoxXYWH | None = None
        self._calib_start: float | None = None
        self._calib_samples: list[float] = []
        self._calibrated = False

    def recalibrate(self) -> None:
        self.calibration = LiveCalibration(roll_deg=self.roll_deg)
        self._calib_start = None
        self._calib_samples = []
        self._calibrated = False
        self.feedback.reset()

    def _update_calibration(self, kp: Keypoints2D | None, t: float) -> float | None:
        if self._calibrated:
            return None
        if self._calib_start is None:
            self._calib_start = t
        if kp is not None:
            ratio = shoulder_ear_ratio(kp)
            if ratio is not None:
                self._calib_samples.append(ratio)
        elapsed = t - self._calib_start
        n = len(self._calib_samples)
        if elapsed >= self.calibration_s and n >= self.min_calibration_samples:
            # p90 = relaxed shoulders (far from the ears); same choice and
            # reasoning as metrics/posture.py compute_pos05a_series.
            baseline = float(np.percentile(self._calib_samples, 90))
            self.calibration = LiveCalibration(roll_deg=self.roll_deg, shrug_baseline=baseline)
            self._calibrated = True
            return None
        time_part = min(1.0, elapsed / self.calibration_s) if self.calibration_s > 0 else 1.0
        sample_part = min(1.0, n / self.min_calibration_samples)
        return min(time_part, sample_part)

    def process(self, image_rgb: np.ndarray, t: float) -> LiveFrameResult:
        h, w = image_rgb.shape[:2]
        xy, conf = self.estimator.estimate(image_rgb, self._box)
        valid = conf >= self.confidence_threshold

        if int(valid.sum()) < self.min_valid_joints:
            self._box = None
            self.filter.reset()
            kp = None
        else:
            smoothed = self.filter(xy / h, t, mask=valid) * h
            kp = Keypoints2D(joint_names=self.estimator.joint_names, xy=smoothed, valid=valid)
            self._box = box_from_keypoints(xy, valid, (w, h))

        progress = self._update_calibration(kp, t)
        values = compute_live_metrics(kp, self.view, self.calibration) if kp is not None else {}
        feedback = None
        if progress is None and kp is not None:
            feedback = self.feedback.update(values, t)
        elif progress is None:
            feedback = self.feedback.update_absent(t)
        return LiveFrameResult(
            t=t, keypoints=kp, box=self._box, values=values, feedback=feedback, calibration_progress=progress
        )
