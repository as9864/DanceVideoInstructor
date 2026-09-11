"""PoseSequence — the perception/analysis boundary (ARCHITECTURE.md §4.2, §1 P1).

This is the one contract the two layers share. Perception (models, GPU) writes
it; analysis (geometry, metrics — pure functions) only ever reads it. Keeping
this single crossing point is what lets the pose backend change without
touching a single metric.
"""

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from dvi.schemas._version import SCHEMA_VERSION
from dvi.schemas.frames import Frame, Role
from dvi.schemas.skeleton import SkeletonSpec


class Provenance(BaseModel):
    """Which model produced this pose. Required — without it, metric comparisons
    across sessions/backends are meaningless (ARCHITECTURE.md §4.2)."""

    backend_name: str
    backend_version: str
    config_hash: str


class BodyScale(BaseModel):
    """Normalization reference for de-scaling distance metrics (ARCHITECTURE.md §5.5).

    At least one of the two must be known; metrics normalize by whichever is
    available (torso_length preferred — more robust to camera-frame scale
    ambiguity than a full-height estimate).
    """

    torso_length_m: float | None = None  # pelvis -> neck distance
    height_estimate_m: float | None = None

    @model_validator(mode="after")
    def _at_least_one_reference(self) -> "BodyScale":
        if self.torso_length_m is None and self.height_estimate_m is None:
            raise ValueError("BodyScale needs torso_length_m or height_estimate_m")
        return self


class PoseSequence(BaseModel):
    schema_version: str = SCHEMA_VERSION
    subject_id: str
    role: Role | None = None
    frame: Frame
    fps: float
    t: np.ndarray  # (T,) seconds
    joints: np.ndarray  # (T, J, 3)
    confidence: np.ndarray  # (T, J) in [0, 1]
    valid: np.ndarray  # (T, J) — confidence gating + interpolation state
    skeleton: SkeletonSpec
    scale: BodyScale
    provenance: Provenance

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("fps")
    @classmethod
    def _fps_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"fps must be positive, got {v}")
        return v

    @field_validator("t")
    @classmethod
    def _t_is_1d_and_monotonic(cls, v: np.ndarray) -> np.ndarray:
        arr = np.asarray(v, dtype=float)
        if arr.ndim != 1:
            raise ValueError(f"t must be 1D (T,), got shape {arr.shape}")
        if arr.shape[0] > 1 and np.any(np.diff(arr) <= 0):
            raise ValueError("t must be strictly increasing")
        return arr

    @model_validator(mode="after")
    def _check_array_shapes(self) -> "PoseSequence":
        t_len = self.t.shape[0]
        j = self.skeleton.n_joints

        expected = {
            "joints": (t_len, j, 3),
            "confidence": (t_len, j),
            "valid": (t_len, j),
        }
        for field_name, expected_shape in expected.items():
            actual_shape = getattr(self, field_name).shape
            if actual_shape != expected_shape:
                raise ValueError(
                    f"{field_name}.shape={actual_shape} does not match expected {expected_shape} "
                    f"(T={t_len} from t, J={j} from skeleton)"
                )

        if self.confidence.size and (self.confidence.min() < 0 or self.confidence.max() > 1):
            raise ValueError("confidence must be within [0, 1]")

        return self

    def require_frame(self, *allowed: Frame) -> None:
        """Runtime guard for the WORLD_G/BODY contract (ARCHITECTURE.md §4.1).

        Posture/axis metrics must call this before computing anything —
        a metric computed in CAMERA frame is silently wrong, not just imprecise.
        """
        if self.frame not in allowed:
            raise ValueError(
                f"PoseSequence.frame={self.frame!r} not in required {tuple(allowed)!r}"
            )
