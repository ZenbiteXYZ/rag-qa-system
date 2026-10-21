# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

ragqa is a RAG question-answering service over versioned `.md` and `.docx` documents, and a lab for comparing chunking strategies and query-side techniques (rewriting, HyDE, Self-Query, hybrid search, reranking) on a hand-written golden set. The design lives in `docs/` (ADR.md, ARCHITECTURE.md, ML_DESIGN.md); read the relevant ADR before changing a decision. The README status section is out of date: ingestion and the document API are implemented, question answering is not.

## Commands

```bash
uv sync --extra dev                      # note: plain `uv sync` / `uv remove` uninstalls the dev extra (pre-commit)
uv run pytest tests/unit/test_chunkers.py::test_locate_skips_a_repeat_inside_the_previous_chunk
uv run pytest -m "not integration"       # what CI runs for unit tests
make check                               # ruff + ruff format --check + mypy + pytest
make format                              # ruff --fix and ruff format
make up PROFILES="ml obs"                # Compose; needs deploy/.env (copy deploy/.env.example). Never read .env
uv run uvicorn --factory ragqa.api.main:create_app --reload   # needs RAGQA_POSTGRES_DSN, RAGQA_RABBITMQ_URL
uv run python -m ragqa.worker            # the ingestion worker
uv run alembic -c alembic.ini revision --autogenerate -m "..."   # dev revisions
```

Integration tests (`-m integration`) skip themselves unless their env vars are set: `RAGQA_TEST_POSTGRES_DSN` (the database name must contain "test", the schema is reset), `RAGQA_TEST_QDRANT_URL`, `RAGQA_TEST_RABBITMQ_URL`. The usual way is throwaway containers (postgres:17.11, qdrant/qdrant:v1.19.2, rabbitmq:4.3.6) on spare ports. Image and action versions are pinned on purpose.

The pre-commit hook runs ruff, mypy and the whole pytest suite on the working tree. Untracked files are always visible to it, while unstaged changes to tracked files are stashed, so a partial commit can fail on tests that import untracked code. Check what is staged before committing.

## Architecture

Two processes share one image and one composition root: the API (`ragqa.api`) and the worker (`ragqa.worker`). `ragqa/wiring.py` (`Backends`) builds the Postgres registry, Qdrant chunk store, file store, TEI embedder and the parser and chunker registries from `Settings` (env vars `RAGQA_*`). Tests swap fields on it.

**Ingestion** (`ingestion/service.py`): API saves the raw file to a shared volume, commits document, version and job rows in Postgres, then publishes a RabbitMQ message. The worker runs `IngestionService.ingest`: parse, chunk, embed, upsert to Qdrant with `is_latest=false`, `Registry.complete_build` (Postgres, may move `documents.latest_version_id`), then flip `is_latest` in Qdrant (ADR-0014). Delivery is at-least-once, so every step must be repeatable: point ids are deterministic (uuid5 of version, index, chunk), and `complete_build` returns True whenever the version is the latest. `PermanentIngestionError` fails a job at once; any other exception is retried with backoff, then dead-lettered (`worker/consumer.py`, `worker/topology.py`). A maintenance loop in the worker republishes jobs stuck in `queued` and finishes deletions. Postgres is the source of truth for "latest"; the Qdrant payload is a copy.

**Document model** (ADR-0021): a parser returns a `ParsedDocument` with one normalized text (NFC, `\n`) and spans into it. Chunks, sections, code blocks and golden-set evidence all use character offsets into that same string, so a chunker must return exact spans (`check_chunks` enforces it). The format is an open string key resolved by a parser registry, not an enum.

**Indexes** (ADR-0013, ADR-0018): one Qdrant collection per (dense model, chunking strategy and params), with a named `dense` vector and a server-side BM25 `bm25` sparse vector, both defined at creation. Postgres `indexes` has a surrogate bigint id that every foreign key and point id uses; `name` is only a handle. Chunkers are looked up by `indexes.chunking_strategy` in `ingestion/chunkers/` (LangChain splitter adapters). Token counts use the embedder's tokenizer.

**Query side** is contracts only so far: `domain/pipeline.py` (`PipelineState`, frozen, one writer per field), `query/stages/base.py` (`Stage`), `query/runner.py` (stub), `query/retriever.py`, `llm/types.py` (`LLMClient`). The pipeline is a fixed sequence of stages switched by config (ADR-0010); an agent mode is deferred to protocol v2 (ADR-0022). History is `ChatMessage` values including tool calls; stages that only need text use `text_turns`.

**LLM access** goes through the LiteLLM proxy over the OpenAI-compatible API (`llm/client.py`, ADR-0016). The app holds no provider keys. `model` is a proxy alias from `deploy/litellm/config.yaml`.

**Schema** is in code: models in `storage/postgres/models/`, migrations inside the package (`storage/postgres/migrations/`, so the image carries them). Compose runs a `migrate` service before the API and worker. An integration test runs `alembic check`, so a model change without a migration fails.

**Observability**: structlog JSON logs (`get_logger`), context bound per request and job, `traced_stage` and `llm_call` in `observability/tracing.py` for MLflow spans and Prometheus metrics, readiness checks in `api/readiness.py`. Use them instead of ad hoc logging or timing.

## Project rules

- Evaluation protocol v1 (ADR-0019) is frozen: metrics, the 42-question split, k=3 / 0.8 coverage, pinned generator, judge and embedder revision. Changes need a new protocol version and ADR, never a silent edit.
- By default the owner writes the query stages and runner, the LiteLLM config, the prompts and the golden questions (the global rules on the protected core apply). Chunkers and `llm/client.py` were also theirs until they asked for them to be written. Infrastructure, storage, parsers, observability and the API are fine to write.
- Do not use em or en dashes in any text.
