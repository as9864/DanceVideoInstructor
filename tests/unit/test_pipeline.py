from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dvi.pipeline.artifacts import ArtifactStore, hash_config
from dvi.pipeline.batch_runner import BatchRunner
from dvi.pipeline.stage import PipelineContext


@dataclass
class CountingStage:
    """Minimal Stage implementation (structural typing — no base class needed)."""

    name: str
    causal: bool = True
    lookahead: int = 0
    cache_key_fields: list[str] = field(default_factory=list)
    call_count: int = 0

    def run(self, data: Any, ctx: PipelineContext) -> Any:
        self.call_count += 1
        return data + 1


def make_ctx(tmp_path: Path, config: dict | None = None) -> PipelineContext:
    config = config or {"foo": 1}
    return PipelineContext(
        config=config,
        input_hash="input-abc",
        config_hash=hash_config(config),
        cache_dir=tmp_path,
    )


class TestBatchRunnerWithoutCache:
    def test_runs_stages_in_order(self, tmp_path):
        stages = [CountingStage("a"), CountingStage("b"), CountingStage("c")]
        runner = BatchRunner(stages, store=None)
        result = runner.run(0, make_ctx(tmp_path))
        assert result == 3
        assert all(s.call_count == 1 for s in stages)


class TestBatchRunnerWithCache:
    def test_second_run_hits_cache(self, tmp_path):
        store = ArtifactStore(tmp_path / "cache")
        stage = CountingStage("a")
        ctx = make_ctx(tmp_path)

        result1 = BatchRunner([stage], store=store).run(0, ctx)
        result2 = BatchRunner([CountingStage("a")], store=store).run(0, ctx)

        assert result1 == result2 == 1
        assert stage.call_count == 1  # second runner's stage never had to run

    def test_different_config_misses_cache(self, tmp_path):
        store = ArtifactStore(tmp_path / "cache")
        stage_config_dependent = CountingStage("a", cache_key_fields=["foo"])

        ctx1 = make_ctx(tmp_path, {"foo": 1})
        ctx2 = make_ctx(tmp_path, {"foo": 2})

        BatchRunner([stage_config_dependent], store=store).run(0, ctx1)
        BatchRunner([stage_config_dependent], store=store).run(0, ctx2)

        assert stage_config_dependent.call_count == 2

    def test_irrelevant_config_change_still_hits_cache(self, tmp_path):
        store = ArtifactStore(tmp_path / "cache")
        # cache_key_fields=[] means this stage doesn't care about any config field
        stage = CountingStage("a", cache_key_fields=[])

        ctx1 = make_ctx(tmp_path, {"foo": 1})
        ctx2 = make_ctx(tmp_path, {"foo": 2})

        BatchRunner([stage], store=store).run(0, ctx1)
        BatchRunner([stage], store=store).run(0, ctx2)

        assert stage.call_count == 1


class TestArtifactStore:
    def test_save_then_load_roundtrip(self, tmp_path):
        store = ArtifactStore(tmp_path)
        store.save("key1", {"a": 1, "b": [1, 2, 3]})
        assert store.has("key1")
        assert store.load("key1") == {"a": 1, "b": [1, 2, 3]}

    def test_missing_key_reports_absent(self, tmp_path):
        store = ArtifactStore(tmp_path)
        assert not store.has("nope")
