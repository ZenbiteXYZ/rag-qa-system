"""Indexes: one Qdrant collection per (dense model, chunking strategy and params) (ADR-0013)."""

from fastapi import APIRouter, HTTPException

from ragqa.api.deps import ResourcesDep, SettingsDep
from ragqa.api.schemas import IndexCreate
from ragqa.domain.indexes import Index, IndexConfig

router = APIRouter(tags=["indexes"])


@router.get("/indexes")
async def list_indexes(resources: ResourcesDep) -> list[Index]:
    return await resources.backends.registry.list_indexes()


@router.post("/indexes", status_code=201)
async def create_index(body: IndexCreate, resources: ResourcesDep, settings: SettingsDep) -> Index:
    backends = resources.backends
    if body.dense_model != settings.embedder_model:
        raise HTTPException(
            422,
            f"this deployment embeds with {settings.embedder_model!r}, not {body.dense_model!r}",
        )
    if body.chunking_strategy not in backends.chunkers.strategies:
        known = ", ".join(backends.chunkers.strategies) or "none registered"
        raise HTTPException(
            422, f"unknown chunking strategy {body.chunking_strategy!r} (known: {known})"
        )
    # The vector size is a property of the model, so measure it instead of trusting the client.
    probe = await backends.embedder.embed_passages(["dimension probe"])
    config = IndexConfig(**body.model_dump(), dense_vector_size=len(probe[0]))
    index = await backends.registry.create_index(config)
    await backends.chunk_store.ensure_collection(index)
    return index
