"""MetricResult / Finding — the scored output of the analysis layer (ARCHITECTURE.md §4.5).

Everything downstream of here must stay traceable to a metric, a frame, and
a threshold (design principle P4, ARCHITECTURE.md §1) — no black-box scoring.
"""

from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

Severity = Literal["info", "minor", "major", "critical"]


class TargetRange(BaseModel):
    """A style profile's threshold for one metric (ARCHITECTURE.md §6.6)."""

    unit: str
    min: float | None = None
    max: float | None = None
    tolerance: float = 0.0

    @model_validator(mode="after")
    def _has_a_bound(self) -> "TargetRange":
        if self.min is None and self.max is None:
            raise ValueError("TargetRange needs at least one of min/max")
        return self


class MetricResult(BaseModel):
    metric_id: str  # e.g. "POS-01"
    series: np.ndarray  # (T,)
    valid: np.ndarray  # (T,) bool
    coverage: float  # fraction of `valid` True — low coverage means "withhold conclusions"
    target: TargetRange
    aggregate: dict[str, float]  # mean/p50/p95/time_out_of_range/...
    causal: bool

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def _check(self) -> "MetricResult":
        if self.series.shape != self.valid.shape:
            raise ValueError(f"series.shape={self.series.shape} != valid.shape={self.valid.shape}")
        if self.series.ndim != 1:
            raise ValueError(f"series must be 1D (T,), got shape {self.series.shape}")
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError(f"coverage must be in [0, 1], got {self.coverage}")
        return self


class TimeSpan(BaseModel):
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int

    @model_validator(mode="after")
    def _end_after_start(self) -> "TimeSpan":
        if self.end_s <= self.start_s:
            raise ValueError(f"end_s ({self.end_s}) must be after start_s ({self.start_s})")
        if self.end_frame < self.start_frame:
            raise ValueError(f"end_frame ({self.end_frame}) before start_frame ({self.start_frame})")
        return self


class Evidence(BaseModel):
    """What a Finding's severity is grounded in — the representative worst frame,
    not the whole span (ARCHITECTURE.md §7.2: "대표 프레임(최대 이탈 지점)")."""

    representative_frame: int
    representative_t_s: float
    measured_value: float
    threshold: float | None = None
    graph_crop: Path | None = None


class Finding(BaseModel):
    finding_id: str
    metric_id: str
    span: TimeSpan
    severity: Severity
    evidence: Evidence
    message_ko: str
    confidence: float  # 0..1

    @model_validator(mode="after")
    def _confidence_in_range(self) -> "Finding":
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        return self
