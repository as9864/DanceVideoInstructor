"""Stage protocol and pipeline context (ARCHITECTURE.md §3.1).

`causal` and `lookahead` are the enforcement point for design principle P5:
a stage declares up front what it needs, so swapping the batch runner for
the (future) stream runner is a config change, not a rewrite.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class PipelineContext:
    config: dict[str, Any]
    input_hash: str  # identifies the source data (e.g. sha256 of the input video)
    config_hash: str  # identifies the full run config (ARCHITECTURE.md §4.6 AnalysisReport.config_hash)
    cache_dir: Path


@runtime_checkable
class Stage(Protocol):
    name: str
    causal: bool
    lookahead: int  # future frames required if non-causal; 0 for causal stages
    cache_key_fields: list[str]  # subset of ctx.config this stage's cache depends on

    def run(self, data: Any, ctx: PipelineContext) -> Any: ...


@dataclass
class FunctionStage:
    """Wrap a plain function as a Stage — enough for stages simple enough that
    a dedicated class would just be ceremony (e.g. Ingest, which is one call
    into VideoReader)."""

    name: str
    fn: Any  # Callable[[Any, PipelineContext], Any]
    causal: bool = True
    lookahead: int = 0
    cache_key_fields: list[str] = field(default_factory=list)

    def run(self, data: Any, ctx: PipelineContext) -> Any:
        return self.fn(data, ctx)
