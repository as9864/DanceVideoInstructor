"""Synthetic/geometric validation of the data contracts (PROJECT.md §7 V1).

These don't test any dance metric — they test that the schemas actually
enforce the shape/frame contracts ARCHITECTURE.md §4 promises, since every
later metric depends on that being true.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from dvi.schemas import (
    BodyScale,
    Evidence,
    FeatureSeries,
    Finding,
    Frame,
    MetricResult,
    PoseSequence,
    Provenance,
    ScoreBreakdown,
    SkeletonSpec,
    TargetRange,
    TimeSpan,
)
from dvi.schemas.skeleton import DVI_CANONICAL_V1


def make_pose_sequence(t_len: int = 10, frame: Frame = Frame.WORLD_G) -> PoseSequence:
    j = DVI_CANONICAL_V1.n_joints
    return PoseSequence(
        subject_id="s1",
        frame=frame,
        fps=30.0,
        t=np.arange(t_len, dtype=float) / 30.0,
        joints=np.zeros((t_len, j, 3)),
        confidence=np.full((t_len, j), 0.9),
        valid=np.ones((t_len, j), dtype=bool),
        skeleton=DVI_CANONICAL_V1,
        scale=BodyScale(torso_length_m=0.5),
        provenance=Provenance(backend_name="synthetic", backend_version="0", config_hash="abc"),
    )


class TestPoseSequence:
    def test_valid_construction(self):
        pose = make_pose_sequence()
        assert pose.joints.shape == (10, DVI_CANONICAL_V1.n_joints, 3)

    def test_joints_shape_mismatch_raises(self):
        pose_kwargs = make_pose_sequence().model_dump()
        pose_kwargs["joints"] = np.zeros((10, DVI_CANONICAL_V1.n_joints - 1, 3))
        with pytest.raises(ValidationError):
            PoseSequence(**{**pose_kwargs, "skeleton": DVI_CANONICAL_V1, "scale": BodyScale(torso_length_m=0.5), "provenance": Provenance(backend_name="x", backend_version="0", config_hash="x")})

    def test_confidence_out_of_range_raises(self):
        pose = make_pose_sequence()
        bad_conf = pose.confidence.copy()
        bad_conf[0, 0] = 1.5
        with pytest.raises(ValidationError):
            PoseSequence(**{**pose.model_dump(exclude={"confidence"}), "confidence": bad_conf, "skeleton": pose.skeleton, "scale": pose.scale, "provenance": pose.provenance})

    def test_t_must_be_strictly_increasing(self):
        pose = make_pose_sequence()
        bad_t = pose.t.copy()
        bad_t[1] = bad_t[0]
        with pytest.raises(ValidationError):
            PoseSequence(**{**pose.model_dump(exclude={"t"}), "t": bad_t, "skeleton": pose.skeleton, "scale": pose.scale, "provenance": pose.provenance})

    def test_require_frame_passes_for_allowed(self):
        pose = make_pose_sequence(frame=Frame.WORLD_G)
        pose.require_frame(Frame.WORLD_G, Frame.BODY)  # must not raise

    def test_require_frame_raises_for_disallowed(self):
        pose = make_pose_sequence(frame=Frame.CAMERA)
        with pytest.raises(ValueError, match="not in required"):
            pose.require_frame(Frame.WORLD_G, Frame.BODY)


class TestSkeletonSpec:
    def test_default_canonical_skeleton(self):
        assert DVI_CANONICAL_V1.n_joints == 23
        assert DVI_CANONICAL_V1.is_derived("spine_mid")
        assert not DVI_CANONICAL_V1.is_derived("pelvis")

    def test_unknown_joint_rejected(self):
        with pytest.raises(ValidationError):
            SkeletonSpec(name="bad", joint_names=("pelvis", "not_a_real_joint"))

    def test_derived_must_be_subset_of_joints(self):
        with pytest.raises(ValidationError):
            SkeletonSpec(name="bad", joint_names=("pelvis", "neck"), derived_joints=frozenset({"spine_mid"}))


class TestBodyScale:
    def test_requires_at_least_one_reference(self):
        with pytest.raises(ValidationError):
            BodyScale()

    def test_torso_length_alone_is_enough(self):
        BodyScale(torso_length_m=0.5)


class TestFeatureSeries:
    def test_valid_construction(self):
        t = np.arange(5, dtype=float)
        fs = FeatureSeries(
            subject_id="s1",
            t=t,
            features={"torso_pitch_deg": np.zeros(5)},
            valid={"torso_pitch_deg": np.ones(5, dtype=bool)},
        )
        assert fs.features["torso_pitch_deg"].shape == (5,)

    def test_feature_length_mismatch_raises(self):
        with pytest.raises(ValidationError):
            FeatureSeries(
                subject_id="s1",
                t=np.arange(5, dtype=float),
                features={"torso_pitch_deg": np.zeros(4)},
                valid={"torso_pitch_deg": np.ones(4, dtype=bool)},
            )

    def test_missing_valid_mask_raises(self):
        with pytest.raises(ValidationError):
            FeatureSeries(
                subject_id="s1",
                t=np.arange(5, dtype=float),
                features={"torso_pitch_deg": np.zeros(5)},
                valid={},
            )


class TestMetricResultAndFinding:
    def test_metric_result_shape_mismatch_raises(self):
        with pytest.raises(ValidationError):
            MetricResult(
                metric_id="POS-01",
                series=np.zeros(10),
                valid=np.ones(9, dtype=bool),
                coverage=1.0,
                target=TargetRange(unit="deg", max=5.0),
                aggregate={"mean": 0.0},
                causal=True,
            )

    def test_target_range_requires_a_bound(self):
        with pytest.raises(ValidationError):
            TargetRange(unit="deg")

    def test_timespan_end_before_start_raises(self):
        with pytest.raises(ValidationError):
            TimeSpan(start_s=1.0, end_s=0.5, start_frame=30, end_frame=15)

    def test_finding_confidence_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            Finding(
                finding_id="f1",
                metric_id="POS-03",
                span=TimeSpan(start_s=0.0, end_s=1.0, start_frame=0, end_frame=30),
                severity="major",
                evidence=Evidence(representative_frame=15, representative_t_s=0.5, measured_value=14.0),
                message_ko="테스트",
                confidence=1.5,
            )


class TestScoreBreakdown:
    def test_mismatched_keys_raise(self):
        with pytest.raises(ValidationError):
            ScoreBreakdown(overall=80.0, categories={"posture": 80.0}, coverage={"axis": 1.0})

    def test_valid_breakdown(self):
        sb = ScoreBreakdown(overall=80.0, categories={"posture": 80.0}, coverage={"posture": 0.95})
        assert sb.overall == 80.0
