# ragqa

A RAG question-answering service over versioned documents, and a lab for comparing retrieval techniques.

Users upload documents (`.md`, `.docx`), the service keeps every version of each document, and answers questions grounded in the right version with citations. Chunking strategies and query-side techniques (query rewriting, HyDE, Self-Query, hybrid search, reranking) are switchable by config and measured on a hand-written golden question set.

## Status

Implemented: document upload and versioning, index creation, and ingestion (parse, chunk, embed, store) through a RabbitMQ worker with retries and a dead-letter queue, plus structured logs, tracing and metrics. Two chunking strategies exist (`fixed-size`, `recursive`). Stored conversations can be read, but nothing writes them yet.

Not implemented yet: question answering (the query pipeline has contracts only), the evaluation runner and the golden question set. Nothing has been evaluated. Agent mode is planned for evaluation protocol v2 (ADR-0022).

## Planned stack

FastAPI, Postgres (registry), Qdrant (dense + BM25 sparse vectors), RabbitMQ with `aio-pika` (ingestion jobs), Hugging Face Text Embeddings Inference with `Snowflake/snowflake-arctic-embed-m-v2.0` (embeddings), LangChain text splitters and the embedder's tokenizer (chunking), LiteLLM proxy (LLM access), MLflow (traces and evaluation runs).

## Quick start

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and Docker.

```bash
# Development environment
uv sync --extra dev
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy

# Run the API and the worker locally (need RAGQA_POSTGRES_DSN and RAGQA_RABBITMQ_URL,
# plus a database that has the migrations applied: python -m ragqa.storage.postgres.migrate)
uv run uvicorn --factory ragqa.api.main:create_app --reload
uv run python -m ragqa.worker
```

Integration tests (`uv run pytest -m integration`) skip themselves unless `RAGQA_TEST_POSTGRES_DSN` (a database whose name contains "test"), `RAGQA_TEST_QDRANT_URL` and `RAGQA_TEST_RABBITMQ_URL` are set.

```bash
# Services with Docker Compose
cp deploy/.env.example deploy/.env   # then fill in the passwords
make up                               # default services
make up PROFILES="ml obs"             # plus TEI, LiteLLM and the monitoring stack
```

The API listens on `http://localhost:8000` (generated docs at `/docs`). MLflow is on port 5000, the RabbitMQ management UI on 15672, Grafana on 3000 and Prometheus on 9090. The first start with the `ml` profile downloads the embedding model, and the first ingestion downloads its tokenizer (about 17 MB) into the `hf_cache` volume.

Typical flow: `POST /indexes` (name, `dense_model`, `chunking_strategy`), then `POST /documents` with a `.md` or `.docx` file, then follow `GET /jobs/{id}` until it is `ready`. The `/docs` page lists every endpoint.

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
src/ragqa/    service code (api, worker, ingestion, query, storage, ...); Alembic migrations are in storage/postgres/migrations/
eval/         golden set, pipeline configs, evaluation runner, reports
deploy/       Dockerfile and Docker Compose
tests/        unit and integration tests
docs/         design documents
```

## License

MIT. See [LICENSE](LICENSE).
