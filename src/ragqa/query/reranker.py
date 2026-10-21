"""Reranker contract and its client for a Text Embeddings Inference cross-encoder (ADR-0023)."""

from collections.abc import Sequence
from typing import Protocol

import httpx

from ragqa.embeddings.tei import EmbedderError, EmbedderUnavailable
from ragqa.observability.metrics import record_dependency_error


class Reranker(Protocol):
    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        """A relevance score per text, aligned with `texts`. Higher is more relevant."""
        ...


class TeiReranker:
    def __init__(self, http: httpx.AsyncClient, base_url: str, *, batch_size: int = 32) -> None:
        self._http = http
        self._url = base_url.rstrip("/") + "/rerank"
        self._batch_size = batch_size

    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        scores = [0.0] * len(texts)
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            for index, score in await self._post(query, batch):
                scores[start + index] = score
        return scores

    async def _post(self, query: str, batch: list[str]) -> list[tuple[int, float]]:
        try:
            response = await self._http.post(
                self._url, json={"query": query, "texts": batch, "raw_scores": False}
            )
        except httpx.HTTPError as exc:
            record_dependency_error("reranker")
            raise EmbedderUnavailable(f"reranker not reachable: {type(exc).__name__}") from exc
        if response.status_code >= 500 or response.status_code == 429:
            record_dependency_error("reranker")
            raise EmbedderUnavailable(f"reranker returned {response.status_code}")
        if response.status_code >= 400:
            raise EmbedderError(f"reranker rejected the input: {response.status_code}")
        data = response.json()
        if not isinstance(data, list) or len(data) != len(batch):
            raise EmbedderError(f"expected {len(batch)} scores, got {type(data).__name__}")
        return [(int(d["index"]), float(d["score"])) for d in data]
