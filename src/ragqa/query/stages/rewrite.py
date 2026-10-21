"""Query rewriting: the question plus the conversation become one standalone search query."""

from ragqa.domain.conversation import ChatMessage
from ragqa.domain.pipeline import PipelineConfig, PipelineState
from ragqa.llm.types import LLMClient, LLMError
from ragqa.query import prompts
from ragqa.query.stages.common import model_for, recent_text_turns, with_call


class RewriteStage:
    name = "rewrite"
    reads = ("question", "history")
    writes = ("rewritten_query",)
    optional = True

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def enabled(self, config: PipelineConfig) -> bool:
        return config.rewrite.enabled

    async def run(self, state: PipelineState) -> PipelineState:
        messages = [
            ChatMessage(role="system", content=prompts.REWRITE_SYSTEM),
            *recent_text_turns(state),
            ChatMessage(role="user", content=state.question),
        ]
        completion = await self._llm.complete(
            messages,
            model=model_for(state.config.rewrite.model, state),
            max_tokens=200,
            reasoning_effort=state.config.reasoning_effort,
        )
        query = completion.text.strip().strip("\"'")
        if not query:
            raise LLMError("the rewrite is empty")
        return state.model_copy(
            update={"rewritten_query": query, "llm_calls": with_call(state, self.name, completion)}
        )
