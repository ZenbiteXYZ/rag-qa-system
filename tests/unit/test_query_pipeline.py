import uuid
from datetime import UTC, datetime

import pytest

from ragqa.domain.conversation import ChatMessage, ToolCall
from ragqa.domain.pipeline import ContextConfig, PipelineConfig, PipelineState
from ragqa.domain.retrieval import KnownDocument, KnownValues, MetadataFilter
from ragqa.llm.types import LLMUnavailable
from ragqa.query import prompts
from ragqa.query.runner import QueryDeps, StageContractError, build_stages, run_pipeline, run_stage
from ragqa.query.stages.context import ContextStage
from ragqa.query.stages.generate import GenerateStage, parse_citations
from ragqa.query.stages.hyde import HydeStage
from ragqa.query.stages.rerank import RerankStage
from ragqa.query.stages.retrieve import RetrieveStage
from ragqa.query.stages.rewrite import RewriteStage
from ragqa.query.stages.self_query import SelfQueryStage
from tests.query_fakes import (
    FakeLLM,
    FakeReranker,
    FakeRetriever,
    chunk,
    no_known_values,
    self_query_json,
    state,
)


def config(**sections: object) -> PipelineConfig:
    return PipelineConfig.model_validate({"default_model": "haiku-5.5", **sections})


# --- rewrite ---


async def test_rewrite_uses_the_text_of_the_conversation_only() -> None:
    llm = FakeLLM(texts=["kubectl drain flags for the second node"])
    call = ToolCall(id="c1", name="search_docs")
    history = (
        ChatMessage(role="user", content="How do I drain a node?"),
        ChatMessage(role="assistant", tool_calls=(call,)),
        ChatMessage(role="tool", content="SECRET TOOL RESULT", tool_call_id="c1"),
        ChatMessage(role="assistant", content="Use kubectl drain."),
    )
    start = state("and the flags?", history=history, config=config(rewrite={"enabled": True}))

    result = await RewriteStage(llm).run(start)

    assert result.rewritten_query == "kubectl drain flags for the second node"
    sent = llm.calls[0]["messages"]
    assert [m.role for m in sent] == ["system", "user", "assistant", "user"]
    assert "SECRET TOOL RESULT" not in " ".join(m.content for m in sent)
    assert llm.calls[0]["effort"] == "medium"
    assert result.llm_calls[0].stage == "rewrite" and result.llm_calls[0].prompt_tokens == 100


async def test_rewrite_respects_the_history_limit() -> None:
    llm = FakeLLM(texts=["q"])
    history = tuple(ChatMessage(role="user", content=f"m{i}") for i in range(6))
    start = state(history=history, config=config(history={"max_messages": 2}))

    await RewriteStage(llm).run(start)

    contents = [m.content for m in llm.calls[0]["messages"]]
    assert "m5" in contents and "m4" in contents and "m3" not in contents


# --- self-query ---

PEOPLE = KnownValues(
    authors=("I. Petrov", "A. Ivanova"),
    doc_types=("task", "concept"),
    tags=("ops", "security"),
    documents=(
        KnownDocument(uuid.UUID(int=7), "Drain a Node", None),
        KnownDocument(uuid.UUID(int=8), "Old Guide", datetime(2026, 1, 1, tzinfo=UTC)),
    ),
)


async def people() -> KnownValues:
    return PEOPLE


def sq_state(**sections: object) -> PipelineState:
    return state("tasks by Petrov about drain", config=config(self_query={"enabled": True}))


async def test_self_query_substitutes_and_records_the_outcome() -> None:
    llm = FakeLLM(
        structured=[
            self_query_json(author="Petrov", doc_type="task", tags=["ops"], semantic_query="drain")
        ]
    )

    result = await SelfQueryStage(llm, people).run(sq_state())

    assert result.semantic_query == "drain"
    assert result.parsed_filters is not None and result.parsed_filters.author == "Petrov"
    assert result.resolved_filters is not None
    assert result.resolved_filters.author == "I. Petrov"
    assert result.resolved_filters.doc_type == "task" and result.resolved_filters.tags == ("ops",)
    outcomes = {(s.field, s.outcome) for s in result.substitutions}
    assert outcomes == {("author", "substituted"), ("doc_type", "exact"), ("tags", "exact")}
    assert any("replaced by 'I. Petrov'" in w for w in result.warnings)
    assert result.outcome is None


async def test_self_query_asks_when_no_candidate_is_clearly_best() -> None:
    llm = FakeLLM(structured=[self_query_json(author="Petro")])
    people_two = KnownValues(authors=("I. Petrov", "M. Petrenko"))

    async def known() -> KnownValues:
        return people_two

    result = await SelfQueryStage(llm, known).run(sq_state())

    assert result.outcome == "needs_clarification"
    assert result.clarification is not None
    assert result.clarification.field == "author"
    assert set(result.clarification.candidates) == {"I. Petrov", "M. Petrenko"}


async def test_self_query_drops_values_that_match_nothing() -> None:
    llm = FakeLLM(structured=[self_query_json(author="Zhang")])

    result = await SelfQueryStage(llm, people).run(sq_state())

    assert result.resolved_filters == MetadataFilter()
    assert result.substitutions[0].outcome == "unresolved"
    assert any("dropped" in w for w in result.warnings)


async def test_self_query_resolves_a_document_and_detects_a_deleted_one() -> None:
    live = await SelfQueryStage(
        FakeLLM(structured=[self_query_json(document_title="drain a node")]), people
    ).run(sq_state())
    assert live.resolved_filters is not None
    assert live.resolved_filters.document_ids == (uuid.UUID(int=7),)
    assert live.resolved_document_title == "Drain a Node"

    gone = await SelfQueryStage(
        FakeLLM(structured=[self_query_json(document_title="Old Guide")]), people
    ).run(sq_state())
    assert gone.outcome == "document_deleted"
    assert gone.deleted_document is not None and gone.deleted_document.title == "Old Guide"


async def test_self_query_keeps_dates_and_falls_back_to_the_question() -> None:
    llm = FakeLLM(structured=[self_query_json(date_from="2026-01-01", semantic_query="")])

    result = await SelfQueryStage(llm, no_known_values).run(sq_state())

    assert result.semantic_query == "tasks by Petrov about drain"
    assert result.resolved_filters is not None
    assert str(result.resolved_filters.date_from) == "2026-01-01"


# --- hyde and retrieve ---


async def test_hyde_text_feeds_the_dense_branch_only() -> None:
    llm = FakeLLM(texts=["To drain a node, run kubectl drain."])
    start = state(config=config(hyde={"enabled": True}, retrieve={"mode": "hybrid"}))
    start = await HydeStage(llm).run(start)
    retriever = FakeRetriever([chunk(0)])

    await RetrieveStage(retriever).run(start)

    query = retriever.queries[0]
    assert query.dense_text == "To drain a node, run kubectl drain."
    assert query.sparse_text == "How do I drain a node?"
    assert query.mode == "hybrid" and query.top_k == 10


async def test_retrieve_text_priority_and_filter_merge() -> None:
    retriever = FakeRetriever([])
    start = state(
        user_filters=MetadataFilter(author="A. Ivanova"),
        version=2,
        rewritten_query="rewritten",
        semantic_query="semantic",
        resolved_filters=MetadataFilter(author="I. Petrov", doc_type="task"),
    )

    await RetrieveStage(retriever).run(start)

    query = retriever.queries[0]
    assert query.dense_text == query.sparse_text == "semantic"
    assert query.filters.author == "A. Ivanova"  # the user's filter wins
    assert query.filters.doc_type == "task"
    assert query.version == 2


# --- rerank and context ---


async def test_rerank_reorders_and_keeps_top_n() -> None:
    reranker = FakeReranker([0.1, 0.9, 0.5])
    start = state(
        config=config(rerank={"enabled": True, "top_n": 2}),
    ).model_copy(update={"candidates": (chunk(0), chunk(1), chunk(2))})

    result = await RerankStage(reranker).run(start)

    assert [c.chunk_index for c in result.ranked] == [1, 2]
    assert [c.score for c in result.ranked] == [0.9, 0.5]
    assert reranker.calls[0][0] == "How do I drain a node?"


async def test_context_respects_the_token_budget_in_rank_order() -> None:
    candidates = tuple(chunk(i, token_count=40) for i in range(5))
    start = state(config=config(context={"max_tokens": 100})).model_copy(
        update={"candidates": candidates}
    )

    result = await ContextStage(FakeRetriever(), None).run(start)

    assert [c.chunk_index for c in result.context_chunks] == [0, 1]


async def test_context_prefers_ranked_over_candidates_and_adds_neighbors_in_order() -> None:
    version = uuid.UUID(int=5)
    hit = chunk(3, version_id=version)
    before = chunk(2, version_id=version, neighbor=True)
    after = chunk(4, version_id=version, neighbor=True)
    start = state(config=config(context={"neighbors": 1})).model_copy(
        update={"candidates": (chunk(9),), "ranked": (hit,)}
    )

    result = await ContextStage(FakeRetriever(neighbors=[after, before]), None).run(start)

    assert [(c.chunk_index, c.is_neighbor) for c in result.context_chunks] == [
        (2, True),
        (3, False),
        (4, True),
    ]


async def test_context_counts_missing_token_counts_with_the_counter() -> None:
    class Words:
        def count(self, text: str) -> int:
            return len(text.split())

    start = state(config=config(context={"max_tokens": 5})).model_copy(
        update={
            "candidates": (
                chunk(0, "one two three", token_count=None),
                chunk(1, "four five six", token_count=None),
            )
        }
    )

    result = await ContextStage(FakeRetriever(), lambda index: Words()).run(start)

    assert [c.chunk_index for c in result.context_chunks] == [0]
    estimated = await ContextStage(FakeRetriever(), None).run(start)
    assert "estimated" in estimated.warnings[-1]


# --- generate ---


def with_context(n: int = 2, **sections: object) -> PipelineState:
    return state(config=config(**sections)).model_copy(
        update={"context_chunks": tuple(chunk(i) for i in range(n))}
    )


async def test_generate_cites_numbered_excerpts_and_flags_invalid_numbers() -> None:
    llm = FakeLLM(texts=["Use drain [1]. It evicts pods [2][1][7]."])

    result = await GenerateStage(llm).run(with_context(2))

    assert result.answer == "Use drain [1]. It evicts pods [2][1][7]."
    assert result.citation_numbers == (1, 2, 7)
    assert [c.chunk_index for c in result.citations] == [0, 1]
    assert result.abstained is False
    assert result.prompt is not None and "[1] Drain > Drain a node > Options" in result.prompt
    user = llm.calls[0]["messages"][-1].content
    assert user.endswith("Question: How do I drain a node?")


async def test_generate_detects_the_refusal_sentence() -> None:
    llm = FakeLLM(texts=[prompts.REFUSAL])

    result = await GenerateStage(llm).run(with_context(1))

    assert result.abstained is True and result.citations == ()


async def test_generate_without_context_refuses_without_calling_the_llm() -> None:
    llm = FakeLLM()

    result = await GenerateStage(llm).run(state())

    assert result.answer == prompts.REFUSAL and result.abstained
    assert llm.calls == []
    assert "no context" in result.warnings[-1]


async def test_closed_book_generation_has_no_excerpts() -> None:
    llm = FakeLLM(texts=["From memory."])
    start = state(config=config(retrieve={"enabled": False}))

    result = await GenerateStage(llm).run(start)

    assert result.answer == "From memory." and result.abstained is False
    sent = llm.calls[0]["messages"]
    assert sent[0].content == prompts.CLOSED_BOOK_SYSTEM
    assert sent[-1].content == "How do I drain a node?"


async def test_generate_includes_recent_conversation() -> None:
    llm = FakeLLM(texts=["ok"])
    history = (
        ChatMessage(role="user", content="earlier question"),
        ChatMessage(role="assistant", content="earlier answer"),
    )

    await GenerateStage(llm).run(with_context(1).model_copy(update={"history": history}))

    roles = [m.role for m in llm.calls[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


async def test_generate_stream_yields_deltas_then_the_state() -> None:
    llm = FakeLLM(texts=["Use drain [1]."])

    events = [e async for e in GenerateStage(llm).stream(with_context(1))]

    assert [type(e).__name__ for e in events[:-1]] == ["TextDelta"] * 3
    final = events[-1]
    assert isinstance(final, PipelineState) and final.answer == "Use drain [1]."
    assert final.citation_numbers == (1,)


def test_parse_citations_ignores_non_citation_brackets() -> None:
    numbers, citations = parse_citations("a[x] b[1] c[ 2 ] d[1]", [chunk(0)])
    assert numbers == (1,) and len(citations) == 1


# --- runner ---


def deps(llm: FakeLLM, retriever: FakeRetriever, reranker: FakeReranker | None = None) -> QueryDeps:
    return QueryDeps(
        llm=llm,
        retriever=retriever,
        known_values=people,
        token_counter=None,
        reranker=reranker,
    )


async def test_baseline_runs_retrieve_context_generate_only() -> None:
    llm, retriever = FakeLLM(texts=["Use drain [1]."]), FakeRetriever([chunk(0), chunk(1)])

    result = await run_pipeline(build_stages(deps(llm, retriever)), state())

    assert result.outcome == "answer" and result.citation_numbers == (1,)
    assert [t.stage for t in result.timings] == ["retrieve", "context", "generate"]
    assert [c.stage for c in result.llm_calls] == ["generate"]
    assert len(retriever.queries) == 1


async def test_all_techniques_run_in_order() -> None:
    llm = FakeLLM(
        texts=["standalone question", "hypothetical passage", "Answer [1]."],
        structured=[self_query_json(doc_type="task", semantic_query="drain node")],
    )
    retriever = FakeRetriever([chunk(0), chunk(1)])
    full = config(
        rewrite={"enabled": True},
        self_query={"enabled": True},
        hyde={"enabled": True},
        rerank={"enabled": True, "top_n": 1},
        retrieve={"mode": "hybrid"},
    )

    result = await run_pipeline(
        build_stages(deps(llm, retriever, FakeReranker([0.2, 0.8]))), state(config=full)
    )

    assert [t.stage for t in result.timings] == [
        "rewrite",
        "self_query",
        "hyde",
        "retrieve",
        "rerank",
        "context",
        "generate",
    ]
    assert result.rewritten_query == "standalone question"
    assert [c.chunk_index for c in result.ranked] == [1]
    assert [c.chunk_index for c in result.context_chunks] == [1]
    assert len(result.llm_calls) == 4


async def test_a_clarification_ends_the_run_before_retrieval() -> None:
    llm = FakeLLM(structured=[self_query_json(author="Petro")])
    retriever = FakeRetriever([chunk(0)])

    async def known() -> KnownValues:
        return KnownValues(authors=("I. Petrov", "M. Petrenko"))

    stages = build_stages(QueryDeps(llm, retriever, known, None))
    result = await run_pipeline(stages, state(config=config(self_query={"enabled": True})))

    assert result.outcome == "needs_clarification"
    assert retriever.queries == [] and result.answer is None


async def test_a_failing_optional_stage_is_skipped_and_counted() -> None:
    llm = FakeLLM(texts=["unused"], fail=LLMUnavailable("down"))
    retriever = FakeRetriever([chunk(0)])
    stages = build_stages(deps(llm, retriever))
    # Rewrite fails, then generate would fail too, so test the optional stage on its own.
    result = await run_stage(stages[0], state(config=config(rewrite={"enabled": True})))

    assert result.degraded == ("rewrite",)
    assert "rewrite failed and was skipped: LLMUnavailable" in result.warnings
    assert result.rewritten_query is None


async def test_a_failing_required_stage_fails_the_request() -> None:
    llm = FakeLLM(fail=LLMUnavailable("down"))
    stages = build_stages(deps(llm, FakeRetriever([chunk(0)])))

    with pytest.raises(LLMUnavailable):
        await run_pipeline(stages, state())


async def test_a_stage_that_writes_an_undeclared_field_is_a_bug() -> None:
    class Sneaky:
        name = "sneaky"
        reads = ("question",)
        writes = ("rewritten_query",)
        optional = False

        def enabled(self, cfg: PipelineConfig) -> bool:
            return True

        async def run(self, s: PipelineState) -> PipelineState:
            return s.model_copy(update={"rewritten_query": "ok", "answer": "oops"})

    with pytest.raises(StageContractError, match="answer"):
        await run_stage(Sneaky(), state())


async def test_retrieval_only_run_stops_after_context() -> None:
    llm, retriever = FakeLLM(), FakeRetriever([chunk(0)])
    cfg = config(generate={"enabled": False})

    result = await run_pipeline(build_stages(deps(llm, retriever)), state(config=cfg))

    assert result.answer is None and result.outcome is None
    assert len(result.context_chunks) == 1 and llm.calls == []


def test_context_config_defaults() -> None:
    assert ContextConfig().neighbors == 0
