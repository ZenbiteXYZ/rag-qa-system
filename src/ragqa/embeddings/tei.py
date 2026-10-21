"""Embedder interface and its client for Text Embeddings Inference (ADR-0012).

The base model expects `query: ` in front of queries and nothing in front of passages,
and TEI does not add it (ADR-0012), so the client does.
"""

from collections.abc import Sequence
from typing import Protocol

import httpx

from ragqa.observability.metrics import record_dependency_error


class EmbedderError(Exception):
    """The embedder rejected the input, for example a text longer than the model allows."""


class EmbedderUnavailable(EmbedderError):
    """The embedder could not be reached or failed on its side. Worth retrying."""


class Embedder(Protocol):
    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


class TeiEmbedder:
    def __init__(
        self,
        http: httpx.AsyncClient,
        base_url: str,
        *,
        query_prefix: str = "query: ",
        passage_prefix: str = "",
        batch_size: int = 32,
    ) -> None:
        self._http = http
        self._url = base_url.rstrip("/") + "/embed"
        self._query_prefix = query_prefix
        self._passage_prefix = passage_prefix
        self._batch_size = batch_size

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed([self._query_prefix + t for t in texts])

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return await self._embed([self._passage_prefix + t for t in texts])

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors.extend(await self._post(batch))
        if vectors and len({len(v) for v in vectors}) != 1:
            raise EmbedderError("the embedder returned vectors of different sizes")
        return vectors

    async def _post(self, batch: list[str]) -> list[list[float]]:
        try:
            response = await self._http.post(self._url, json={"inputs": batch})
        except httpx.HTTPError as exc:
            record_dependency_error("embedder")
            raise EmbedderUnavailable(f"embedder not reachable: {type(exc).__name__}") from exc
        if response.status_code >= 500 or response.status_code == 429:
            record_dependency_error("embedder")
            raise EmbedderUnavailable(f"embedder returned {response.status_code}")
        if response.status_code >= 400:
            raise EmbedderError(
                f"embedder rejected the input: {response.status_code} {response.text[:200]}"
            )
        data = response.json()
        if not isinstance(data, list) or len(data) != len(batch):
            raise EmbedderError(f"expected {len(batch)} vectors, got {type(data).__name__}")
        return [[float(x) for x in vector] for vector in data]
