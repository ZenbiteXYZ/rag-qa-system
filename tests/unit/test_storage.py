import uuid
from datetime import date
from pathlib import Path

import httpx
import pytest

from ragqa.domain.documents import Chunk, DocumentMetadata, EmbeddedChunk
from ragqa.domain.indexes import IndexConfig, collection_for
from ragqa.embeddings.tei import EmbedderError, EmbedderUnavailable, TeiEmbedder
from ragqa.storage.files.store import FileStore, safe_filename
from ragqa.storage.qdrant.store import VersionRef, chunk_payload, point_id

# --- Index config


def test_index_name_rules_and_collection() -> None:
    config = IndexConfig(
        name="arcticm2_fixed_512o64",
        dense_model="Snowflake/snowflake-arctic-embed-m-v2.0",
        dense_vector_size=768,
        chunking_strategy="fixed",
        chunking_params={"size": 512, "overlap": 64},
    )

    assert config.dense_distance == "cosine"
    assert config.sparse_model == "Qdrant/bm25"
    assert collection_for(config.name) == "chunks_arcticm2_fixed_512o64"
    for bad in ("Upper", "-leading", "has space", "x" * 64):
        with pytest.raises(ValueError):
            IndexConfig(name=bad, dense_model="m", dense_vector_size=8, chunking_strategy="s")


# --- Qdrant payload and ids


def test_point_id_is_deterministic_and_distinct() -> None:
    version = uuid.uuid4()

    assert point_id(version, 1, 0) == point_id(version, 1, 0)
    assert len({point_id(version, 1, 0), point_id(version, 2, 0), point_id(version, 1, 1)}) == 3


def test_chunk_payload() -> None:
    ref = VersionRef(
        document_id=uuid.uuid4(),
        version_id=uuid.uuid4(),
        version=3,
        title="Pods",
        metadata=DocumentMetadata(doc_type="concept", tags=("pods",), doc_date=date(2026, 3, 1)),
    )
    chunk = Chunk(
        chunk_index=2, start=10, end=40, text="A Pod", section_path=("Pods",), token_count=2
    )

    payload = chunk_payload(ref, EmbeddedChunk(chunk=chunk, dense=(0.1, 0.2)))

    assert payload["is_latest"] is False  # flipped only after Postgres commits
    assert payload["document_id"] == str(ref.document_id)
    assert payload["version"] == 3
    assert payload["doc_date"] == "2026-03-01T00:00:00Z"
    assert payload["section_path"] == ["Pods"]
    assert (payload["start"], payload["end"], payload["chunk_index"]) == (10, 40, 2)
    assert payload["text"] == "A Pod"


# --- File store


@pytest.mark.parametrize(
    "name,expected",
    [
        ("guide.md", "guide.md"),
        ("../../etc/passwd", "passwd"),
        ("my file (1).docx", "my_file_1_.docx"),
        ("...", "file"),
    ],
)
def test_safe_filename(name: str, expected: str) -> None:
    assert safe_filename(name) == expected


async def test_file_store_round_trip_and_delete(tmp_path: Path) -> None:
    store = FileStore(tmp_path)
    document_id, version_id = uuid.uuid4(), uuid.uuid4()
    relative = store.path_for(document_id, version_id, "guide.md")

    await store.save(relative, b"# Guide")

    assert await store.read(relative) == b"# Guide"
    assert not list((tmp_path / str(document_id) / str(version_id)).glob(".tmp-*"))
    await store.delete_document(document_id)
    await store.delete_document(document_id)  # safe to repeat
    assert not (tmp_path / str(document_id)).exists()


async def test_file_store_rejects_paths_outside_the_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        await FileStore(tmp_path).read("../secret")


# --- TEI embedder


def _embedder(handler: httpx.MockTransport, batch_size: int = 2) -> TeiEmbedder:
    http = httpx.AsyncClient(transport=handler)
    return TeiEmbedder(http, "http://tei", batch_size=batch_size)


async def test_query_prefix_and_batching() -> None:
    seen: list[list[str]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        inputs = httpx.Response(200, content=request.content).json()["inputs"]
        seen.append(inputs)
        return httpx.Response(200, json=[[0.1, 0.2] for _ in inputs])

    embedder = _embedder(httpx.MockTransport(handle))

    vectors = await embedder.embed_queries(["a", "b", "c"])
    await embedder.embed_passages(["p"])

    assert vectors == [[0.1, 0.2]] * 3
    assert seen == [["query: a", "query: b"], ["query: c"], ["p"]]


@pytest.mark.parametrize(
    "status,error", [(503, EmbedderUnavailable), (429, EmbedderUnavailable), (413, EmbedderError)]
)
async def test_error_classes(status: int, error: type[Exception]) -> None:
    embedder = _embedder(httpx.MockTransport(lambda r: httpx.Response(status, text="nope")))

    with pytest.raises(error):
        await embedder.embed_passages(["x"])


async def test_unreachable_and_bad_shape() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(EmbedderUnavailable):
        await _embedder(httpx.MockTransport(refuse)).embed_passages(["x"])
    with pytest.raises(EmbedderError, match="expected 1 vectors"):
        await _embedder(httpx.MockTransport(lambda r: httpx.Response(200, json=[]))).embed_passages(
            ["x"]
        )
