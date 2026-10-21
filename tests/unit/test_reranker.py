import json

import httpx
import pytest

from ragqa.embeddings.tei import EmbedderError, EmbedderUnavailable
from ragqa.query.reranker import TeiReranker


def reranker(handler: object, batch_size: int = 32) -> TeiReranker:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]
    return TeiReranker(http, "http://rerank/", batch_size=batch_size)


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
