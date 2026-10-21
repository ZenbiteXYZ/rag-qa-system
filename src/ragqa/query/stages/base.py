"""Stage contract (ADR-0010)."""

from typing import Protocol

from ragqa.domain.pipeline import PipelineConfig, PipelineState


class Stage(Protocol):
    name: str  # span and metric name, for example "rewrite"
    # State field names, recorded on the span by `traced_stage`. The runner may also check
    # that `run` changed nothing outside `writes`.
    reads: tuple[str, ...]
    writes: tuple[str, ...]

    def enabled(self, config: PipelineConfig) -> bool: ...

    async def run(self, state: PipelineState) -> PipelineState: ...
