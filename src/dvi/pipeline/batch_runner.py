"""Batch pipeline runner (ARCHITECTURE.md §2, §3.1).

Runs every stage regardless of `causal` — that's the stream runner's job to
restrict (ARCHITECTURE.md §11). This runner's only responsibility is
sequencing stages and caching their output per-stage.
"""

import hashlib
from collections.abc import Sequence
from typing import Any

from dvi.pipeline.artifacts import ArtifactStore, hash_config
from dvi.pipeline.stage import PipelineContext, Stage


class BatchRunner:
    def __init__(self, stages: Sequence[Stage], store: ArtifactStore | None = None):
        self.stages = list(stages)
        self.store = store

    def run(self, initial_data: Any, ctx: PipelineContext) -> Any:
        data = initial_data
        for stage in self.stages:
            key = self._cache_key(stage, ctx)
            if self.store is not None and self.store.has(key):
                data = self.store.load(key)
                continue
            data = stage.run(data, ctx)
            if self.store is not None:
                self.store.save(key, data)
        return data

    @staticmethod
    def _cache_key(stage: Stage, ctx: PipelineContext) -> str:
        relevant_config = {k: ctx.config.get(k) for k in stage.cache_key_fields}
        raw = f"{stage.name}:{ctx.input_hash}:{hash_config(relevant_config)}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
