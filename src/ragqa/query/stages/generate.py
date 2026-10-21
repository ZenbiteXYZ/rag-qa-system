"""Generation: the answer from the question, the conversation and the numbered context, or from
the model's own knowledge when retrieval is off (the closed-book control)."""

import re
from collections.abc import AsyncIterator, Sequence

from ragqa.domain.conversation import ChatMessage
from ragqa.domain.pipeline import Citation, PipelineConfig, PipelineState
from ragqa.domain.retrieval import RetrievedChunk
from ragqa.llm.types import Completion, LLMClient, TextDelta
from ragqa.query import prompts
from ragqa.query.stages.common import model_for, recent_text_turns, with_call

_CITATION = re.compile(r"\[(\d+)\]")
_QUOTE_CHARS = 300


def parse_citations(
    answer: str, context: Sequence[RetrievedChunk]
) -> tuple[tuple[int, ...], tuple[Citation, ...]]:
    """Every [n] in the answer (once each, in order), and the Citations for those that point
    at an excerpt that exists."""
    numbers: list[int] = []
    for match in _CITATION.finditer(answer):
        number = int(match.group(1))
        if number not in numbers:
            numbers.append(number)
    citations = tuple(
        Citation(
            document_id=chunk.document_id,
            version=chunk.version,
            title=chunk.title,
            section_path=chunk.section_path,
            chunk_index=chunk.chunk_index,
            quote=chunk.text[:_QUOTE_CHARS],
        )
        for number in numbers
        if 1 <= number <= len(context)
        for chunk in [context[number - 1]]
    )
    return tuple(numbers), citations


class GenerateStage:
    name = "generate"
    reads = ("question", "history", "context_chunks", "config")
    writes = ("prompt", "answer", "abstained", "citation_numbers", "citations")
    optional = False

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def enabled(self, config: PipelineConfig) -> bool:
        return config.generate.enabled

    def _messages(self, state: PipelineState) -> tuple[list[ChatMessage], str]:
        closed_book = not state.config.retrieve.enabled
        user = (
            state.question
            if closed_book
            else prompts.grounded_user_message(state.question, state.context_chunks)
        )
        system = prompts.CLOSED_BOOK_SYSTEM if closed_book else prompts.GROUNDED_SYSTEM
        messages = [
            ChatMessage(role="system", content=system),
            *recent_text_turns(state),
            ChatMessage(role="user", content=user),
        ]
        return messages, user

    def _finish(
        self, state: PipelineState, prompt: str, completion: Completion | None, text: str
    ) -> PipelineState:
        numbers, citations = parse_citations(text, state.context_chunks)
        update: dict[str, object] = {
            "prompt": prompt,
            "answer": text,
            "abstained": text.strip().startswith(prompts.REFUSAL),
            "citation_numbers": numbers,
            "citations": citations,
        }
        if completion is not None:
            update["llm_calls"] = with_call(state, self.name, completion)
        return state.model_copy(update=update)

    def _no_context(self, state: PipelineState) -> bool:
        return state.config.retrieve.enabled and not state.context_chunks

    async def run(self, state: PipelineState) -> PipelineState:
        messages, user = self._messages(state)
        if self._no_context(state):
            # Nothing was retrieved (for example the filters excluded everything).
            warned = state.model_copy(
                update={"warnings": (*state.warnings, "no context was retrieved")}
            )
            return self._finish(warned, user, None, prompts.REFUSAL)
        config = state.config.generate
        completion = await self._llm.complete(
            messages,
            model=model_for(config.model, state),
            max_tokens=config.max_tokens,
            reasoning_effort=state.config.reasoning_effort,
        )
        return self._finish(state, user, completion, completion.text.strip())

    async def stream(self, state: PipelineState) -> AsyncIterator[TextDelta | PipelineState]:
        """Text deltas, then the finished state."""
        messages, user = self._messages(state)
        if self._no_context(state):
            yield TextDelta(text=prompts.REFUSAL)
            warned = state.model_copy(
                update={"warnings": (*state.warnings, "no context was retrieved")}
            )
            yield self._finish(warned, user, None, prompts.REFUSAL)
            return
        config = state.config.generate
        async for event in self._llm.stream(
            messages,
            model=model_for(config.model, state),
            max_tokens=config.max_tokens,
            reasoning_effort=state.config.reasoning_effort,
        ):
            if isinstance(event, TextDelta):
                yield event
            else:
                yield self._finish(state, user, event, event.text.strip())
