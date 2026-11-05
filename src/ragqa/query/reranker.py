"""Reranker contract and its two clients (ADR-0023): a Text Embeddings Inference cross-encoder
that runs locally, and a hosted rerank API (OpenRouter or the LiteLLM proxy)."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol

import httpx

from ragqa.embeddings.tei import EmbedderError, EmbedderUnavailable
from ragqa.observability.metrics import record_dependency_error

if TYPE_CHECKING:
    from ragqa.settings import Settings


class Reranker(Protocol):
    @property
    def name(self) -> str:
        """Which reranker this is, for example `tei:BAAI/bge-reranker-v2-m3`. Recorded in the
        pipeline state, because it is a pinned model of every run that reranks."""
        ...

    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        """A relevance score per text, aligned with `texts`. Higher is more relevant."""
        ...


class TeiReranker:
    def __init__(
        self, http: httpx.AsyncClient, base_url: str, *, name: str, batch_size: int = 32
    ) -> None:
        self._http = http
        self._url = base_url.rstrip("/") + "/rerank"
        self._batch_size = batch_size
        self.name = name

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


class ApiReranker:
    """A hosted rerank API with the Cohere-style request and response, as served by OpenRouter
    (`POST <base>/rerank`) and by the LiteLLM proxy (`POST <base>/v1/rerank`).

    The key goes into the request as a bearer token. Pointing `base_url` at the LiteLLM proxy
    keeps provider keys out of the app (ADR-0016); pointing it at OpenRouter puts that key in
    this process's environment instead.
    """

    def __init__(
        self,
        http: httpx.AsyncClient,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        name: str,
    ) -> None:
        self._http = http
        self._url = base_url.rstrip("/") + "/rerank"
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.name = name

    async def rerank(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        try:
            response = await self._http.post(
                self._url,
                headers=self._headers,
                json={
                    "model": self._model,
                    "query": query,
                    "documents": list(texts),
                    "top_n": len(texts),
                },
            )
        except httpx.HTTPError as exc:
            record_dependency_error("reranker")
            raise EmbedderUnavailable(f"rerank API not reachable: {type(exc).__name__}") from exc
        if response.status_code >= 500 or response.status_code == 429:
            record_dependency_error("reranker")
            raise EmbedderUnavailable(f"rerank API returned {response.status_code}")
        if response.status_code >= 400:
            # 401, 402 (no credit) and 404 (unknown model) are not worth retrying.
            raise EmbedderError(f"rerank API rejected the request: {response.status_code}")
        return self._scores(response.json(), len(texts))

    @staticmethod
    def _scores(data: object, count: int) -> list[float]:
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list) or len(results) != count:
            raise EmbedderError(f"expected {count} rerank results")
        scores: list[float | None] = [None] * count
        for item in results:
            index = item["index"]
            if not isinstance(index, int) or not 0 <= index < count:
                raise EmbedderError(f"rerank result with index {index!r}")
            scores[index] = float(item["relevance_score"])
        if any(score is None for score in scores):
            raise EmbedderError("the rerank API did not score every document")
        return [score for score in scores if score is not None]


DEFAULT_LOCAL_MODEL = "BAAI/bge-reranker-v2-m3"


def build_reranker(settings: "Settings", http: httpx.AsyncClient) -> Reranker | None:
    """The reranker the settings ask for, or None when `reranker_url` is not set."""
    if not settings.reranker_url:
        return None
    if settings.reranker_kind == "api":
        assert settings.reranker_model  # checked by Settings
        key = settings.reranker_api_key.get_secret_value() if settings.reranker_api_key else None
        return ApiReranker(
            http,
            settings.reranker_url,
            settings.reranker_model,
            api_key=key or None,
            name=f"api:{settings.reranker_model}",
        )
    model = settings.reranker_model or DEFAULT_LOCAL_MODEL
    return TeiReranker(http, settings.reranker_url, name=f"tei:{model}")
