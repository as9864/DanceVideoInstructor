"""AnalysisReport — the final artifact handed to the presentation layer (ARCHITECTURE.md §4.6)."""

from pathlib import Path

from pydantic import BaseModel, model_validator

from dvi.schemas._version import SCHEMA_VERSION
from dvi.schemas.frames import Role
from dvi.schemas.metrics import Finding, MetricResult


class SourceInfo(BaseModel):
    path: Path
    width: int
    height: int
    fps: float
    duration_s: float
    sha256: str


class ScoreBreakdown(BaseModel):
    """Score is meaningless without coverage next to it (ARCHITECTURE.md §7.1:
    "점수는 반드시 커버리지와 함께 표시한다"). Both dicts share the same keys."""

    overall: float  # 0..100
    categories: dict[str, float]  # e.g. "posture" -> 0..100
    coverage: dict[str, float]  # same keys -> fraction of valid frames behind that score

    @model_validator(mode="after")
    def _check(self) -> "ScoreBreakdown":
        if not (0.0 <= self.overall <= 100.0):
            raise ValueError(f"overall must be in [0, 100], got {self.overall}")
        if set(self.categories) != set(self.coverage):
            raise ValueError(
                f"categories keys {sorted(self.categories)} != coverage keys {sorted(self.coverage)}"
            )
        for name, score in self.categories.items():
            if not (0.0 <= score <= 100.0):
                raise ValueError(f"categories[{name!r}]={score} not in [0, 100]")
        for name, cov in self.coverage.items():
            if not (0.0 <= cov <= 1.0):
                raise ValueError(f"coverage[{name!r}]={cov} not in [0, 1]")
        return self


class SubjectSummary(BaseModel):
    subject_id: str
    role: Role | None = None
    scores: ScoreBreakdown
    metrics: list[MetricResult]


class CoupleSummary(BaseModel):
    scores: ScoreBreakdown
    metrics: list[MetricResult]


class AnalysisReport(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source: SourceInfo
    config_hash: str
    style_profile: str
    subjects: list[SubjectSummary]
    couple: CoupleSummary | None = None
    findings: list[Finding]
    scores: ScoreBreakdown
    artifacts: dict[str, Path]
    warnings: list[str] = []
