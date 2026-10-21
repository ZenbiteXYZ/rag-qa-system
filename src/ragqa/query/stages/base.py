"""Stage contract (ADR-0010)."""

from typing import Protocol

from ragqa.domain.pipeline import PipelineConfig, PipelineState

# Fields every stage may touch besides its declared `writes`: bookkeeping that the runner and
# LLM stages maintain for the trace, the response and the system metrics.
BOOKKEEPING = frozenset({"warnings", "degraded", "timings", "llm_calls"})


class Stage(Protocol):
    # Read-only properties, so implementations can use plain class attributes.
    @property
    def name(self) -> str:  # span and metric name, for example "rewrite"
        ...

    # State field names, recorded on the span by `traced_stage`. The runner checks that `run`
    # changed nothing outside `writes` and `BOOKKEEPING`.
    @property
    def reads(self) -> tuple[str, ...]: ...

    @property
    def writes(self) -> tuple[str, ...]: ...

    # An optional stage that fails is skipped and reported in `state.degraded`; a required one
    # fails the request.
    @property
    def optional(self) -> bool: ...

    def enabled(self, config: PipelineConfig) -> bool: ...

    async def run(self, state: PipelineState) -> PipelineState: ...
