from collections.abc import Mapping
from typing import Any

import pytest
from pydantic import ValidationError

from ragqa.domain.conversation import ChatMessage, ToolCall, text_turns
from ragqa.domain.documents import Chunk, ParsedDocument
from ragqa.domain.indexes import Index
from ragqa.domain.pipeline import Clarification, PipelineConfig, PipelineState
from ragqa.domain.retrieval import RetrievalQuery
from ragqa.ingestion.chunkers import ChunkerRegistry, TokenCounter, UnknownStrategyError


class _WholeDoc:
    strategy = "whole"

    def __init__(self, params: Mapping[str, Any], counter: TokenCounter) -> None:
        self._counter = counter

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        n = self._counter.count(doc.text)
        return [Chunk(start=0, end=len(doc.text), chunk_index=0, text=doc.text, token_count=n)]


class _Words:
    def count(self, text: str) -> int:
        return len(text.split())


def test_registry_builds_registered_strategy() -> None:
    registry = ChunkerRegistry()
    registry.register("whole", _WholeDoc)
    chunker = registry.build("whole", {}, _Words())
    doc = ParsedDocument(title="t", source_format="md", text="one two three")
    assert chunker.chunk(doc)[0].token_count == 3


def test_registry_rejects_unknown_and_duplicate() -> None:
    registry = ChunkerRegistry()
    registry.register("whole", _WholeDoc)
    with pytest.raises(UnknownStrategyError, match="known: whole"):
        registry.build("semantic", {}, _Words())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("whole", _WholeDoc)


def test_retrieval_query_needs_text_for_each_branch() -> None:
    RetrievalQuery(mode="sparse", sparse_text="kubectl drain")
    with pytest.raises(ValidationError, match="needs sparse_text"):
        RetrievalQuery(mode="hybrid", dense_text="how to drain a node")


def _state() -> PipelineState:
    index = Index(
        id=1,
        name="base",
        collection="chunks_base",
        dense_model="m",
        dense_vector_size=768,
        chunking_strategy="fixed",
    )
    return PipelineState(question="q", index=index, config=PipelineConfig(default_model="m"))


def test_state_outcome_and_immutability() -> None:
    state = _state()
    assert state.outcome is None
    stopped = state.model_copy(
        update={"clarification": Clarification(field="author", value="Ivan", candidates=())}
    )
    assert stopped.outcome == "needs_clarification"
    assert state.outcome is None
    with pytest.raises(ValidationError):
        state.answer = "x"


def test_chat_message_role_rules() -> None:
    call = ToolCall(id="c1", name="search_docs", arguments={"q": "ivan"})
    ChatMessage(role="assistant", tool_calls=(call,))
    ChatMessage(role="tool", content="[]", tool_call_id="c1")
    with pytest.raises(ValidationError, match="tool_call_id"):
        ChatMessage(role="tool", content="[]")
    with pytest.raises(ValidationError, match="only assistant"):
        ChatMessage(role="user", content="hi", tool_calls=(call,))
    with pytest.raises(ValidationError, match="needs content"):
        ChatMessage(role="user")


def test_state_carries_history() -> None:
    history = (
        ChatMessage(role="user", content="How do I drain a node?"),
        ChatMessage(role="assistant", content="Use kubectl drain."),
    )
    state = _state().model_copy(update={"history": history})
    assert [m.role for m in state.history] == ["user", "assistant"]


def test_text_turns_drops_tools() -> None:
    call = ToolCall(id="c1", name="search_docs")
    history = (
        ChatMessage(role="user", content="Who wrote it?"),
        ChatMessage(role="assistant", tool_calls=(call,)),
        ChatMessage(role="tool", content="[]", tool_call_id="c1"),
        ChatMessage(role="assistant", content="Checking.", tool_calls=(call,)),
    )
    result = text_turns(history)
    assert [(m.role, m.content) for m in result] == [
        ("user", "Who wrote it?"),
        ("assistant", "Checking."),
    ]
    assert all(not m.tool_calls for m in result)
