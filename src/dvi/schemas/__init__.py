"""Data contracts (ARCHITECTURE.md §4). Pure Pydantic — no I/O, no model deps (P1/P3)."""

from dvi.schemas._version import SCHEMA_VERSION
from dvi.schemas.features import FeatureSeries
from dvi.schemas.frames import Frame, Role
from dvi.schemas.metrics import Evidence, Finding, MetricResult, Severity, TargetRange, TimeSpan
from dvi.schemas.pose import BodyScale, PoseSequence, Provenance
from dvi.schemas.report import AnalysisReport, CoupleSummary, ScoreBreakdown, SourceInfo, SubjectSummary
from dvi.schemas.skeleton import DVI_CANONICAL_JOINTS, DVI_CANONICAL_V1, SkeletonSpec

__all__ = [
    "SCHEMA_VERSION",
    "Frame",
    "Role",
    "SkeletonSpec",
    "DVI_CANONICAL_JOINTS",
    "DVI_CANONICAL_V1",
    "Provenance",
    "BodyScale",
    "PoseSequence",
    "FeatureSeries",
    "TargetRange",
    "MetricResult",
    "TimeSpan",
    "Evidence",
    "Severity",
    "Finding",
    "SourceInfo",
    "ScoreBreakdown",
    "SubjectSummary",
    "CoupleSummary",
    "AnalysisReport",
]
