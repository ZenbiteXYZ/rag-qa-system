# ragqa

A RAG question-answering service over versioned documents, and a lab for comparing retrieval techniques.

Users upload documents (`.md`, `.docx`), the service keeps every version of each document, and answers questions grounded in the right version with citations. Chunking strategies and query-side techniques (query rewriting, HyDE, Self-Query, hybrid search, reranking) are switchable by config and measured on a hand-written golden question set.

## Status

Infrastructure is in place: API skeleton (`/health`, `/ready`, `/metrics`), ingestion worker with retry and dead-letter queues, structured logs, tracing and metrics. Ingestion and question answering are not implemented yet. Nothing has been evaluated yet.

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

# Run the API locally (needs RAGQA_POSTGRES_DSN and RAGQA_RABBITMQ_URL)
uv run uvicorn --factory ragqa.api.main:create_app --reload
```

```bash
# Services with Docker Compose
cp deploy/.env.example deploy/.env   # then fill in the passwords
make up                               # default services
make up PROFILES="ml obs"             # plus TEI, LiteLLM and the monitoring stack
```

The API listens on `http://localhost:8000` (generated docs at `/docs`). MLflow is on port 5000, the RabbitMQ management UI on 15672, Grafana on 3000 and Prometheus on 9090. The first start with the `ml` profile downloads the embedding model.

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
