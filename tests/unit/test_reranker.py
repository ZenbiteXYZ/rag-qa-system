import json

import httpx
import pytest
from pydantic import SecretStr

from ragqa.embeddings.tei import EmbedderError, EmbedderUnavailable
from ragqa.query.reranker import ApiReranker, TeiReranker, build_reranker
from ragqa.settings import Settings


def reranker(handler: object, batch_size: int = 32) -> TeiReranker:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return TeiReranker(http, "http://rerank/", batch_size=batch_size, name="tei:test")


async def test_scores_are_aligned_with_the_input_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert request.url.path == "/rerank" and body["query"] == "q"
        # TEI answers sorted by score, with the index of each text.
        return httpx.Response(200, json=[{"index": 1, "score": 0.9}, {"index": 0, "score": 0.1}])

    assert await reranker(handler).rerank("q", ["a", "b"]) == [0.1, 0.9]


async def test_texts_are_sent_in_batches() -> None:
    sizes: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        texts = json.loads(request.content)["texts"]
        sizes.append(len(texts))
        return httpx.Response(
            200, json=[{"index": i, "score": float(i)} for i in range(len(texts))]
        )

    scores = await reranker(handler, batch_size=2).rerank("q", ["a", "b", "c"])

    assert sizes == [2, 1] and scores == [0.0, 1.0, 0.0]


@pytest.mark.parametrize(("status", "error"), [(500, EmbedderUnavailable), (400, EmbedderError)])
async def test_errors_are_classified(status: int, error: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status)

    with pytest.raises(error):
        await reranker(handler).rerank("q", ["a"])


# --- hosted rerank API (OpenRouter or the LiteLLM proxy) ---


def api_reranker(handler: object, key: str | None = "secret-key") -> ApiReranker:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return ApiReranker(
        http, "https://openrouter.ai/api/v1/", "cohere/rerank-v3.5", api_key=key, name="api:test"
    )


async def test_api_reranker_sends_the_cohere_style_request_and_orders_scores_by_index() -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        # Results come sorted by relevance, each with the index of its document.
        results = [
            {"index": 2, "relevance_score": 0.9, "document": {"text": "c"}},
            {"index": 0, "relevance_score": 0.5},
            {"index": 1, "relevance_score": 0.1},
        ]
        return httpx.Response(200, json={"results": results, "usage": {"search_units": 1}})

    scores = await api_reranker(handler).rerank("q", ["a", "b", "c"])

    assert scores == [0.5, 0.1, 0.9]
    assert seen["url"] == "https://openrouter.ai/api/v1/rerank"
    assert seen["auth"] == "Bearer secret-key"
    assert seen["body"] == {
        "model": "cohere/rerank-v3.5",
        "query": "q",
        "documents": ["a", "b", "c"],
        "top_n": 3,
    }


async def test_api_reranker_without_a_key_sends_no_authorization() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

    assert await api_reranker(handler, key=None).rerank("q", ["a"]) == [1.0]
    assert await api_reranker(handler, key=None).rerank("q", []) == []


@pytest.mark.parametrize(
    ("status", "error"),
    [
        (500, EmbedderUnavailable),
        (429, EmbedderUnavailable),
        (401, EmbedderError),
        (402, EmbedderError),
    ],
)
async def test_api_reranker_errors_are_classified(status: int, error: type[Exception]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": {"message": "nope"}})

    with pytest.raises(error):
        await api_reranker(handler).rerank("q", ["a"])


async def test_api_reranker_connection_errors_are_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    with pytest.raises(EmbedderUnavailable):
        await api_reranker(handler).rerank("q", ["a"])


@pytest.mark.parametrize(
    ("body", "documents"),
    [
        ({"results": []}, ["a"]),  # wrong number of results
        ({"results": [{"index": 5, "relevance_score": 1.0}]}, ["a"]),  # index out of range
        (
            {
                "results": [
                    {"index": 0, "relevance_score": 1.0},
                    {"index": 0, "relevance_score": 1.0},
                ]
            },
            ["a", "b"],  # one document scored twice, one not at all
        ),
        ({"oops": True}, ["a"]),
    ],
)
async def test_api_reranker_rejects_malformed_responses(
    body: dict[str, object], documents: list[str]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    with pytest.raises(EmbedderError):
        await api_reranker(handler).rerank("q", documents)


# --- choosing the implementation from the settings ---


def settings_with(settings: Settings, **update: object) -> Settings:
    return Settings(**{**settings.model_dump(), **update})  # type: ignore[arg-type]


def test_no_url_means_no_reranker(settings: Settings) -> None:
    http = httpx.AsyncClient()
    assert build_reranker(settings, http) is None


def test_the_local_reranker_is_the_default(settings: Settings) -> None:
    http = httpx.AsyncClient()
    local = build_reranker(settings_with(settings, reranker_url="http://tei-rerank:80"), http)
    assert isinstance(local, TeiReranker) and local.name == "tei:BAAI/bge-reranker-v2-m3"


def test_the_api_reranker_needs_a_model_and_takes_the_key_as_a_secret(settings: Settings) -> None:
    http = httpx.AsyncClient()
    with pytest.raises(ValueError, match="RERANKER_MODEL"):
        settings_with(settings, reranker_kind="api", reranker_url="https://openrouter.ai/api/v1")

    configured = settings_with(
        settings,
        reranker_kind="api",
        reranker_url="https://openrouter.ai/api/v1",
        reranker_model="cohere/rerank-v3.5",
        reranker_api_key=SecretStr("sk-or-test"),
    )
    reranker = build_reranker(configured, http)
    assert isinstance(reranker, ApiReranker) and reranker.name == "api:cohere/rerank-v3.5"
    assert "sk-or-test" not in repr(configured)  # never printed


def test_an_empty_key_from_compose_means_no_key(settings: Settings) -> None:
    http = httpx.AsyncClient()
    configured = settings_with(
        settings,
        reranker_kind="api",
        reranker_url="http://litellm:4000/v1",
        reranker_model="rerank",
        reranker_api_key=SecretStr(""),
    )
    reranker = build_reranker(configured, http)
    assert isinstance(reranker, ApiReranker) and reranker._headers == {}
