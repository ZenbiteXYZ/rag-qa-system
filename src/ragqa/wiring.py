"""The long-lived clients that both the API and the worker use, built from `Settings`."""

from dataclasses import dataclass

import httpx
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncEngine

from ragqa.embeddings.tei import Embedder, TeiEmbedder
from ragqa.ingestion.chunkers import ChunkerRegistry, TokenCounterFactory
from ragqa.ingestion.chunkers import default_registry as default_chunkers
from ragqa.ingestion.parsers import ParserRegistry
from ragqa.ingestion.parsers import default_registry as default_parsers
from ragqa.llm.client import ProxyLLMClient
from ragqa.query.reranker import Reranker, build_reranker
from ragqa.query.retriever import QdrantRetriever
from ragqa.settings import Settings
from ragqa.storage.files.store import FileStore
from ragqa.storage.postgres.registry import Registry, create_session_factory
from ragqa.storage.qdrant.store import ChunkStore


@dataclass
class Backends:
    engine: AsyncEngine
    registry: Registry
    files: FileStore
    qdrant: AsyncQdrantClient
    chunk_store: ChunkStore
    embedder: Embedder
    parsers: ParserRegistry
    chunkers: ChunkerRegistry
    token_counter: TokenCounterFactory | None
    retriever: QdrantRetriever
    llm: ProxyLLMClient
    reranker: Reranker | None
    _embedder_http: httpx.AsyncClient

    @classmethod
    def create(
        cls, settings: Settings, *, token_counter: TokenCounterFactory | None = None
    ) -> "Backends":
        engine, sessions = create_session_factory(settings.postgres_dsn)
        qdrant = AsyncQdrantClient(url=settings.qdrant_url, timeout=60, check_compatibility=False)
        embedder_http = httpx.AsyncClient(timeout=httpx.Timeout(settings.embedder_timeout_s))
        embedder = TeiEmbedder(
            embedder_http,
            settings.embedder_url,
            query_prefix=settings.embedder_query_prefix,
            passage_prefix=settings.embedder_passage_prefix,
            batch_size=settings.embedder_batch_size,
        )
        chunk_store = ChunkStore(qdrant)
        return cls(
            engine=engine,
            registry=Registry(sessions),
            files=FileStore(settings.files_dir),
            qdrant=qdrant,
            chunk_store=chunk_store,
            embedder=embedder,
            parsers=default_parsers(),
            chunkers=default_chunkers(),
            token_counter=token_counter,
            retriever=QdrantRetriever(chunk_store, embedder),
            llm=ProxyLLMClient.from_settings(settings),
            reranker=build_reranker(settings, embedder_http),
            _embedder_http=embedder_http,
        )

    async def close(self) -> None:
        await self.llm.close()
        await self.qdrant.close()
        await self._embedder_http.aclose()
        await self.engine.dispose()
