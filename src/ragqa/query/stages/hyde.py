"""HyDE: a hypothetical answer, embedded in place of the question for the dense branch only
(ADR-0018)."""

from ragqa.domain.conversation import ChatMessage
from ragqa.domain.pipeline import PipelineConfig, PipelineState
from ragqa.llm.types import LLMClient, LLMError
from ragqa.query import prompts
from ragqa.query.stages.common import base_query, model_for, with_call


class HydeStage:
    name = "hyde"
    reads = ("question", "rewritten_query", "semantic_query")
    writes = ("hypothetical_document",)
    optional = True

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def enabled(self, config: PipelineConfig) -> bool:
        return config.hyde.enabled

    async def run(self, state: PipelineState) -> PipelineState:
        messages = [
            ChatMessage(role="system", content=prompts.HYDE_SYSTEM),
            ChatMessage(role="user", content=base_query(state)),
        ]
        completion = await self._llm.complete(
            messages,
            model=model_for(state.config.hyde.model, state),
            max_tokens=300,
            reasoning_effort=state.config.reasoning_effort,
        )
        text = completion.text.strip()
        if not text:
            raise LLMError("the hypothetical document is empty")
        return state.model_copy(
            update={
                "hypothetical_document": text,
                "llm_calls": with_call(state, self.name, completion),
            }
        )
