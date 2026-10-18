# ragqa

A RAG question-answering service over versioned documents, and a lab for comparing retrieval techniques.

Users upload documents (`.md`, `.docx`), the service keeps every version of each document, and answers questions grounded in the right version with citations. Chunking strategies and query-side techniques (query rewriting, HyDE, Self-Query, hybrid search, reranking) are switchable by config and measured on a hand-written golden question set.

## Status

Design is written, implementation has just started. Only a `/health` endpoint exists. The worker, embedding server, LiteLLM proxy and MLflow are not in Docker Compose yet. Nothing has been evaluated yet.

## Planned stack

FastAPI, Postgres (registry), Qdrant (dense + BM25 sparse vectors), RabbitMQ with `aio-pika` (ingestion jobs), Hugging Face Text Embeddings Inference with `Snowflake/snowflake-arctic-embed-m-v2.0` (embeddings), LiteLLM proxy (LLM access), MLflow (traces and evaluation runs).

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
# Development environment
uv sync --extra dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy

# Run the API locally (only /health for now)
uv run uvicorn ragqa.api.main:app --reload
```

```bash
# Services with Docker Compose
cd deploy
cp .env.example .env        # then change the passwords
docker compose up --build
```

The API listens on `http://localhost:8000` (`/health`, and the generated docs at `/docs`).

## Documentation

| Document | Contents |
|----------|----------|
| [User stories](docs/USER_STORIES.md) | Personas, stories with acceptance criteria, MVP cut |
| [User flows](docs/USER_FLOWS.md) | Upload, versioning, question answering, evaluation, deletion |
| [Architecture](docs/ARCHITECTURE.md) | Components, data model, pipelines, API draft, project layout |
| [ADR](docs/ADR.md) | Architecture decision records |
| [ML design](docs/ML_DESIGN.md) | Hypotheses, data, evaluation protocol, experiment plan, risks |

## Project layout

```
src/ragqa/    service code (api, worker, ingestion, query, storage, ...)
eval/         golden set, pipeline configs, evaluation runner, reports
deploy/       Dockerfile and Docker Compose
migrations/   database migrations
tests/        unit and integration tests
docs/         design documents
```

## License

MIT. See [LICENSE](LICENSE).
