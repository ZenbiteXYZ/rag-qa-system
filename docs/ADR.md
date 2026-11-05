# Architecture Decision Records

Format per record: Status, Context, Decision, Consequences. Statuses: Proposed, Accepted, Superseded.
Date of all records below: 2026-10-14.

## ADR-0001: Supported document formats

- **Status:** Accepted
- **Context:** The service should handle "various kinds of documents", but parsing quality differs a lot by format. PDF layout extraction is unreliable and would dominate the early work.
- **Decision:** Support `.md` and `.docx` only. PDF, TXT and HTML are out of scope for v1.
- **Consequences:** Both formats expose heading structure, so one "sections" abstraction can feed structure-aware chunking. Uploads of other types are rejected with a clear error (story A1). PDF can be added later behind the same parser interface.

## ADR-0002: Storage: Qdrant for vectors, Postgres for the registry

- **Status:** Accepted
- **Context:** Chunks and vectors need payload filtering (Self-Query) and possibly sparse plus dense vectors (hybrid search). The system also needs relational data: documents, versions, the `latest` pointer, ingestion status, known metadata values, and later conversations, traces and eval runs. Considered: pgvector only, Qdrant only (registry as payload-only points), Qdrant plus SQLite.
- **Decision:** Qdrant for chunks and vectors. Postgres for the registry and for later conversations, traces and eval runs.
- **Consequences:** Two stateful services in Docker Compose (story F3). Consistency between them must be handled by the ingestion worker: a failed ingestion cleans up partial vectors, and the `latest` pointer moves only after success.

## ADR-0003: Versioning granularity: per file upload

- **Status:** Accepted
- **Context:** Per-section versioning would allow chunk reuse but needs stable chunk ids and chunk-level hashes.
- **Decision:** A version is one uploaded file. A new version is re-parsed and fully re-embedded. Identical content (same hash) creates no new version.
- **Consequences:** Simple model and simple rollback (move the pointer). Re-embedding cost grows with document size. The diff between versions (story B4) is a text diff of the parsed content, not a chunk diff.

## ADR-0004: Interface: API only

- **Status:** Accepted
- **Context:** A UI would add work unrelated to the learning goals.
- **Decision:** FastAPI service only. The auto-generated `/docs` page is the test interface.
- **Consequences:** Clients and any later UI depend on the API contract, so request and response schemas need care from the start.

## ADR-0005: Golden question set is written by hand

- **Status:** Accepted
- **Context:** Gains from HyDE, rewriting and Self-Query cannot be judged without evaluation data. LLM-generated questions tend to copy wording from the source chunk, which favors lexical and dense retrieval and hides the benefit of query-side techniques.
- **Decision:** The author writes the questions by hand (their number and split are fixed in ADR-0019), some with metadata cues ("in the latest guide", "by author X"). Synthetic questions are a supplement only. The set is stored in a file under version control.
- **Consequences:** Small set, so differences between configs must be reported with variance across runs and not as point estimates. Writing the set takes real time.

## ADR-0006: Evaluation corpus: Kubernetes docs

- **Status:** Accepted
- **Context:** The corpus needs Markdown, real version differences, and usable metadata. The author delegated the choice.
- **Decision:** `kubernetes/website`, `content/en/docs`, license CC BY 4.0. Three release branches, `release-1.32`, `release-1.34` and `release-1.36`, two releases apart so that version-sensitive questions have real differences. Use a subset such as `concepts/` and `tasks/`, English only. Front matter provides `title`, `content_type` (concept, task, tutorial, reference), `reviewers` and `weight`.
- **Consequences:** Attribution is required. The corpus has no reliable author field, so author metadata is synthetic where absent (see ADR-0007). The pages use Hugo shortcodes (code samples, includes, headings, feature states), so a preparation step renders them to plain Markdown per release (`ragqa.evaluation.corpus`), records the source commit and a hash of every file in a manifest, and the result is kept outside the repository, read-only. Rendering follows the site's templates as of each release, and a few decorative shortcodes are dropped (listed in ML_DESIGN section 3).

## ADR-0007: Metadata schema

- **Status:** Proposed
- **Context:** Self-Query is only as useful as the metadata it can filter on, and the fields must be ones users mention in questions. Version fields are rarely mentioned and a wrong guess silently returns stale answers.
- **Decision:**
  - Document level: `doc_type`, `tags`, `author`, `doc_date` (date stated by the content), `uploaded_at`, `language`.
  - Chunk level: `section_path` (heading hierarchy), position.
  - System filters, not available to Self-Query: `version`, `is_latest`.
- **Consequences:** The field stays named `author`. The chosen corpus has no reliable author, so where it is absent, synthetic metadata is assigned from a versioned mapping file to test filters and Self-Query. Questions that rely on author filters are therefore synthetic-metadata tests, not tests on natural cues in the content.

## ADR-0008: Self-Query filter value validation

- **Status:** Accepted
- **Context:** The LLM may invent a filter value that does not exist, for example `author="Ivan"` when the stored value is `"I. Petrov"`. Applying it silently returns empty or wrong results.
- **Decision:** Filter values are checked against the known values stored in Postgres. For an unknown value, fuzzy matching finds candidates. If one candidate is clearly best, it is used and the substitution is reported in the trace. Otherwise the system returns the nearest candidates and asks the user to choose.
- **Consequences:** Needs a table of known values per filterable field, kept in sync on ingestion and deletion. The "clearly best" threshold is to be calibrated on real corpus values. With a stateless pipeline, the clarification is a "needs clarification" response listing candidates, and the client resends the request with the chosen value.

## ADR-0009: Models are configurable: local embeddings, API LLM

- **Status:** Accepted
- **Context:** Embeddings are called on every chunk and every query, so local inference saves cost. LLM quality matters for rewriting, HyDE, Self-Query parsing and generation.
- **Decision:** Embedding model and LLM are selected in config files. Embeddings run locally, the LLM is called through an API. Model names and versions are logged per run.
- **Consequences:** Changing the embedding model requires re-indexing (vector dimensions and semantics change), so the model name is stored with the index. The base embedding model is `Snowflake/snowflake-arctic-embed-m-v2.0` (ADR-0012). LLM models are chosen per stage in the LiteLLM proxy config (ADR-0016).

## ADR-0010: Pipeline shape: fixed sequence of stages

- **Status:** Accepted
- **Context:** An agent that chooses its own tool calls is flexible, but a metric change could then come from a different tool path and not from the technique being tested.
- **Decision:** The query pipeline is a fixed sequence of stages, and each technique is switched by config. Wrapping the pipeline as an agent tool is a later experiment. (Scheduled for protocol v2 in ADR-0022.)
- **Consequences:** Effects of single techniques can be attributed, and runs are more repeatable. Less adaptive behavior on complex questions until multi-query or decomposition (story D6) is added as another stage.

## ADR-0011: Ingestion job transport: RabbitMQ

- **Status:** Accepted
- **Context:** The API must hand ingestion work to a separate worker (story F1). Considered: Postgres as a queue (`SELECT ... FOR UPDATE SKIP LOCKED`), Redis with Arq or Celery, RabbitMQ, Kafka. Kafka was rejected as overkill. The author chose RabbitMQ, partly to practice with a real broker.
- **Decision:** RabbitMQ carries ingestion jobs from the API to the worker. Job status (queued, parsing, chunking, embedding, ready, failed, dead) stays in Postgres for story A2. Failed jobs go to a dead-letter queue after 3 retries with backoff. Permanent errors (unparseable file) skip retries.
- **Consequences:** Publishing and the Postgres commit are two writes, so a republisher re-sends jobs stuck in `queued` past a timeout (or an outbox table is used). Delivery is at-least-once, so consumers must be idempotent (see ADR-0014). Ack only after work completes, prefetch 1, and check the consumer ack timeout against the slowest ingestion. Delayed retries need TTL queues or a delay plugin (verify in the docs). The Python client library is `aio-pika` (asyncio, direct control over acks, prefetch and dead-letter exchanges). CPU-bound parsing in the worker runs in an executor. One more container in Compose.

## ADR-0012: Embedder runs as a separate service

- **Status:** Accepted
- **Context:** Embeddings are needed at ingestion (worker) and at query time (API: the query and the HyDE text). Loading the model in both processes duplicates memory and risks a silent model version mismatch between index and query.
- **Decision:** One embedding service is called by both API and worker, behind an `Embedder` interface. An in-process implementation of the same interface is available for development and tests. The server is Hugging Face Text Embeddings Inference (TEI), which does token-based dynamic batching and runs on CPU. The base model is `Snowflake/snowflake-arctic-embed-m-v2.0` (see ADR-0009). Per its model card: 768 dimensions, context window up to 8192 tokens, 74 languages, Apache-2.0 license, Matryoshka (MRL) support with 256 dimensions. Other models can be added as separate indexes and compared (ADR-0013).
- **Consequences:** One model for index and query by construction, and the API stays light. One more container and a network hop on every query. The embedder's model name is part of the index identity (ADR-0013). Many embedding models need different prefixes for queries and passages, so the `Embedder` interface must distinguish the two. For the base model, queries get the prefix `query: ` and documents get none. TEI does not apply it: at startup TEI 1.9.4 logs that it found no Sentence Transformers config for this model, so the embedder client adds `query: ` itself (checked against the running container on 2026-10-19).

## ADR-0013: Qdrant layout: one collection per index config

- **Status:** Accepted
- **Context:** Vector size is fixed per collection and differs by embedding model. Mixing chunking strategies in one collection mixes their results and confounds chunking experiments (story D1).
- **Decision:** One Qdrant collection per index config, defined as (embedding model, chunking strategy and params), named like `chunks_<embedder>_<chunker>_<params>`. The user chooses the target index at upload, and a query selects the index to search. The same document version may be built in several indexes, tracked in a version-to-index build table with its own status.
  - Payload: `document_id`, `version`, `is_latest`, `doc_type`, `tags`, `author`, `doc_date`, `language`, `section_path`, `chunk_index`, `text`.
  - Registry: each index has a surrogate `id` (primary key) and unique `name` and `collection`. Foreign keys use the id.
  - Point id: deterministic UUID v5 from (version id, index id, chunk index). The id, not the name, so a rename keeps point ids.
  - Payload indexes on every filtered field (verify the details in the Qdrant docs).
- **Consequences:** Strategy comparison means ingesting the same corpus into several indexes (more storage and embedding time). Changing the embedding model creates a new index and does not modify an existing one. Deterministic ids make retried upserts overwrite instead of duplicating.

## ADR-0014: Consistency between Postgres and Qdrant: ordered steps with compensation

- **Status:** Accepted
- **Context:** No transaction spans Postgres and Qdrant, so a unit of work protects only the Postgres side. Default queries filter on `is_latest`, which lets us hide unfinished work. Postgres is the source of truth for which version is latest (`documents.latest_version_id`). The Qdrant `is_latest` payload is a denormalized copy kept for fast filtering.
- **Decision:** Ingestion runs in this order:
  1. Write vectors with `is_latest=false`. Default queries cannot see them.
  2. In one Postgres transaction: mark the version ready, move `latest_version_id`, finish the job.
  3. Set `is_latest=true` on the new version's points, then `false` on the old version's points. A brief overlap is preferred over a gap.

  Failure at step 1 or 2: delete points by `version_id` and mark the job failed, or retry (idempotent). Failure at step 3: retry. A reconciler repairs any mismatch between the two stores. Deletion marks the document deleted in Postgres first (hidden), then deletes vectors by filter, then raw files, all idempotent, and keeps a tombstone row (stories A3, A6).
- **Consequences:** Short windows of eventual consistency, in which a query may briefly see both versions of a document. Every step must be idempotent. A reconciler job and an invariant check (exactly one latest version per document in both stores) are needed.

## ADR-0015: Traces and evaluation runs are logged in MLflow

- **Status:** Accepted
- **Context:** Story D7 needs per-request traces with stage inputs, outputs, retrieved chunks, prompts, latency and token cost. The eval runner must read retrieved chunk ids per question programmatically. MLflow Tracing records inputs, outputs, latency and token usage per step, is OpenTelemetry-compatible, has async logging, and supports search through `mlflow.search_traces`. A SQL-backed store (PostgreSQL is supported) gives the full search syntax. MLflow is also already in the author's stack.
- **Decision:** MLflow is the store for request traces (one span per pipeline stage) and for evaluation runs (pipeline config as params, metrics, report and golden set version as artifacts). The MLflow server uses a separate database in the existing Postgres, and a volume for artifacts. The API uses the lightweight `mlflow-tracing` package with async logging. The trace id returned by `POST /query` is the MLflow trace id. Prometheus and Grafana for operational metrics (RPS, failure rate) are deferred and optional.
- **Consequences:** One more container. The MLflow version must be pinned, since the search result schema differs between 2.x and 3.x. Limits on span payload size were not found in the docs I read, so test early with a realistic prompt and chunk list. MLflow is not an operational metrics system.

## ADR-0016: LLM access through a LiteLLM proxy

- **Status:** Accepted
- **Context:** ADR-0009 requires configurable models, and different stages (rewriting, Self-Query parsing, HyDE, generation) may want different models or providers. Provider SDKs differ in API shape.
- **Decision:** Run LiteLLM as a proxy server. The API talks to it through an OpenAI-compatible API, and model names in the pipeline config are aliases defined in the proxy config. Provider keys live only in the proxy's environment.
- **Consequences:** One more container. Switching a stage to another model or provider is a config change, and the application has a single client implementation. Provider keys stay out of the API code and config. The proxy's logging and any MLflow integration are to be checked in the docs before relying on them (ADR-0015 traces come from the API's own spans).

## ADR-0017: Raw uploaded files on a shared volume

- **Status:** Accepted
- **Context:** The API stores uploaded files and the worker reads them for ingestion, so both need access to the same storage. Considered: a Docker volume and an S3-compatible store such as MinIO.
- **Decision:** A Docker volume mounted into both API and worker. File access goes through a `files/` storage module, so the backend can be swapped later.
- **Consequences:** No extra service. Not suited to running API and worker on different hosts. Moving to an object store later means a new implementation of the `files/` module and a migration of existing files.

## ADR-0018: Hybrid retrieval: dense and BM25 sparse vectors in every index

- **Status:** Accepted
- **Context:** Dense embeddings capture meaning but can miss exact tokens such as flag names, identifiers and error codes. Story D5 asks for hybrid search. Qdrant supports named dense and sparse vectors on one point, an IDF modifier for sparse vectors, and a query API that runs several `prefetch` branches and fuses them with Reciprocal Rank Fusion (per the Qdrant docs). Hybrid is a search method that fuses two rankings, not a single vector. The base embedder (ADR-0012) is dense only.
- **Decision:** Every index collection defines a named `dense` vector and a sparse `bm25` vector with the IDF modifier (refines ADR-0013). Each chunk is upserted with both. The retrieve stage has `mode` of `dense`, `sparse` or `hybrid`, where hybrid is one query with a prefetch branch per vector merged by RRF. The evaluation baseline (E2) stays dense only. The dense branch may use the HyDE text, and the sparse branch uses the Self-Query semantic query, else the rewritten query, else the question, and never HyDE (to be tested on the golden set).
- **Consequences:** More storage per chunk and a sparse-vector step at ingestion. Per the docs, server-side BM25 needs Qdrant 1.15.3 or newer and the `k` and `weights` options of RRF need 1.17 or newer, so the pinned Qdrant image must be at least that. Per the Qdrant docs, BM25 inference runs inside any Qdrant instance, including self-hosted, so the client sends raw text and needs no tokenization code or extra dependency. (Dense inference by Qdrant is Cloud-only, which does not matter because TEI produces our dense vectors.) A throwaway spike script (not kept in the repo; Qdrant 1.19.2 self-hosted, qdrant-client 1.19.1) confirmed that upsert and query with `Document(model="Qdrant/bm25")` work without the `avg_len` option, and that a hybrid query with a dense and a BM25 prefetch fused by RRF works. The same spike showed that `update_collection(sparse_vectors_config=...)` on an existing dense-only collection fails with a 400 ("Not existing vector name"), so both vectors must be defined when the collection is created, which is the decision above. BM25 matches tokens in one language, so it gives no cross-lingual retrieval and the dense branch carries it.

## ADR-0019: Evaluation protocol v1

- **Status:** Accepted
- **Context:** The project compares retrieval techniques, so the way results are produced must be fixed before the first experiment and changed only on purpose. The full protocol is in [ML_DESIGN.md](ML_DESIGN.md).
- **Decision:** Protocol version 1:
  - **Golden set:** hand-written, 6 questions in each of 7 categories (42 in total), split 3 dev and 3 test per category with a fixed seed. Evidence is tagged as verbatim quotes that a validator resolves against the parser output (ADR-0005, `ragqa.evaluation.evidence`).
  - **Retrieval metrics:** coverage of the evidence by the top-k chunks with k = 3 and a threshold of 0.8, with sensitivity reports at thresholds 0.5 and 1.0 and at k = 5. Multi-evidence questions average over their required items.
  - **Levels and primary metrics:** retrieval (Coverage@3), filter extraction (exact match), end to end (answer correctness), system (p95 latency and tokens per request). All metrics are per question and averaged over questions.
  - **Comparison:** a baseline (fixed-size chunker, dense only, no query-side techniques), one change at a time, a closed-book control, 2 repetitions per configuration, paired per-question comparison with an interval over questions.
  - **Run records:** every run stores its config, versions, raw per-question outputs and judge outputs as immutable MLflow artifacts, and metrics are computed from them only, so a changed metric can be recomputed without regenerating answers (ML_DESIGN section 6).
  - **Models:** generator `anthropic/claude-haiku-5.5-20261007`, judge `openai/gpt-6-luna-20260922` (reasoning effort low, structured output, seed set), embedder `Snowflake/snowflake-arctic-embed-m-v2.0` at revision `95c2741480856aa9666782eb4afe11959938017f`. The judge must pass a calibration gate against 30 hand-labeled dev answers before its scores are used.
- **Consequences:** Changing metrics, splits, the golden set, the judge or its rubric creates protocol version 2 and a new ADR, and results from different versions are not mixed. The set is pilot-scale: with 21 test questions only large differences (about 0.13 or more in coverage, under an assumed per-question standard deviation of 0.3) can be told apart, so results are exploratory, and an inconclusive result is followed by extending the set under protocol v2. Neither LLM accepts sampling settings through OpenRouter, so run-to-run noise has to be measured by repetition. The judge is a third-party model on the same routing layer as the generator, so provider routing is logged. A cheaper decision-model judge (Jev, called through TypeSafe's Decisions API) was considered and not adopted for now: the whole experiment plan costs about $1 with the chosen judge, the API sits outside LiteLLM and is marked alpha, and it returns no explanation to debug rubric disagreements. It can be revisited as a protocol change. The cheap judge is chosen for cost (roughly $0.12 per evaluation run by my token estimate), and the calibration gate is what justifies trusting it. Using `openrouter/` model names through LiteLLM still has to be verified in the pinned LiteLLM version.

## ADR-0020: Observability stack and Compose profiles

- **Status:** Accepted
- **Context:** ADR-0015 put request traces and evaluation runs in MLflow and deferred Prometheus and Grafana. The author chose the full stack from the start: structured logs, infrastructure spans, metrics and log search. Running everything always would pull large images and the embedding model on every start.
- **Decision:**
  - **Logs:** structlog renders every line as JSON, including stdlib loggers (uvicorn, aio-pika, httpx), with `request_id`, `job_id` and `trace_id` from context and secret redaction. Grafana Alloy reads container logs from the Docker socket and pushes them to Loki.
  - **Traces:** one OpenTelemetry tracer provider shared by MLflow and the FastAPI and httpx auto-instrumentation, so infrastructure spans and pipeline spans (`traced_stage`, `llm_call`) land in one MLflow trace. The worker links its job trace to the API trace through W3C headers on the RabbitMQ message. No OTLP exporter or extra trace backend.
  - **Metrics:** `prometheus-client` on the API (`/metrics`) and on the worker (port 9100). Prometheus also scrapes Qdrant, RabbitMQ and TEI. Grafana has provisioned datasources and one operations dashboard.
  - **Compose profiles:** the default set is Postgres, Qdrant, RabbitMQ, MLflow, API and worker. `ml` adds TEI and LiteLLM, `obs` adds Prometheus, Loki, Alloy and Grafana.
- **Consequences:** Supersedes the deferral in ADR-0015. Four more services with the `obs` profile and their volumes. Alloy needs the Docker socket read-only and SELinux labeling disabled for that container, which gives it access to the Docker API. MLflow and LiteLLM are not scraped for metrics yet. Tracing is off unless `RAGQA_TRACING_ENABLED` is set.

## ADR-0021: Document model: one normalized text with spans, formats through a parser registry

- **Status:** Accepted
- **Context:** Parsers, chunkers, the evidence validator and citations all need to agree on positions in a document. More formats (PDF, TXT, spreadsheets, slides) and possibly other modalities are likely later, and adding one should not change the core types.
- **Decision:**
  - A parser returns a `ParsedDocument`: one string `text` (NFC, `\n` line endings) and spans into it for sections and code blocks. Chunks keep their own source span next to their text, so evidence matching (ADR-0019) and citations use the same offsets.
  - The source format is an open string, the key of a parser in a registry (`Parser` objects declare their format keys and file extensions). The API rejects an upload whose extension has no parser. Adding a format means adding a parser, not editing the model. A closed `Literal` or `Enum` in the domain would force that edit.
  - Code blocks are marked as `code_spans` by the parser (fences in markdown). Chunks carry `has_code`, and chunkers can use `overlaps_code` to avoid splitting a block.
  - The types are frozen Pydantic models with their invariants as validators. `src/ragqa/domain/documents.py`.
- **Consequences:** Formats whose structure is not text (pages, slides, sheets) will need optional location fields on sections, which are backward-compatible additions. Other modalities are handled by parsers that convert to text (OCR or captions for images, transcription with timestamps for audio), so they reuse the pipeline and the evaluation unchanged. Native multimodal embeddings, and a separate code embedding model with its own vector, are not part of protocol v1: they would add vectors per index and need their own questions and a protocol change.

## ADR-0022: Agent mode is planned for protocol v2

- **Status:** Accepted (scheduling decision). The design below is a sketch, to be confirmed in a protocol v2 ADR.
- **Context:** The author sees the service as a chat that can run tools and answer on its own, without a fixed workflow. ADR-0010 chose a fixed pipeline so that a metric change can only come from the technique under test, and listed wrapping the pipeline as an agent tool as a later experiment. Coverage@3 and the other v1 metrics assume one retrieval per question (ADR-0019).
- **Decision:** Protocol v1 and its first experiments run on the fixed pipeline only. Agent mode is added under protocol v2, as a second mode next to the pipeline, which stays the baseline and the control. Sketch:
  - Tools are thin wrappers over existing parts: `search_docs(query, filters)` over the `Retriever`, `list_filter_values(field, prefix)` in place of the fuzzy fallback of ADR-0008, and a document lookup that can report tombstones.
  - A step cap bounds the tool calls per question. Clarification and deleted-document outcomes become normal assistant messages.
  - The model is called through the LiteLLM proxy (ADR-0016), so no provider keys enter the app. LangGraph's ready-made agent is the first candidate. Its dependencies are added only when this ADR is confirmed.
  - The v2 evaluation has to define which retrieved chunks count for coverage when an agent searches several times or not at all, and must report latency and tokens per question, since both rise with every extra call.
- **Consequences:** The types are already prepared: `ChatMessage` carries tool calls and tool results, `LLMClient.complete` takes tool definitions, and history can hold tool turns. The `agent/` package is created then. `POST /query` will need a mode switch and a response shape for answers that are plain assistant messages. Results from the two modes are never mixed in one protocol version.

## ADR-0023: Reranking with a TEI cross-encoder

- **Status:** Accepted
- **Context:** The rerank stage (ADR-0010, ML_DESIGN section 2) reorders the retrieved candidates with a model that reads the query and each candidate together. It was left open which model and how it is served. Options: an LLM through the proxy (no new service, but one more LLM call per request with all candidates in the prompt), or a cross-encoder served by TEI (the serving software of ADR-0012, no LLM tokens).
- **Decision:** A cross-encoder served by a second TEI container, `BAAI/bge-reranker-v2-m3` at revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` (Apache-2.0, multilingual, architecture `XLMRobertaForSequenceClassification` per its Hub config), under its own Compose profile `rerank` because the first start downloads about 2.2 GB. The app calls `POST /rerank` (`ragqa.query.reranker.TeiReranker`, address `RAGQA_RERANKER_URL`). A request that enables reranking on a deployment without a reranker URL is rejected with 422. The reranker is one more pinned model: its name and revision belong in every run record that uses it.
- **Consequences:** Reranking adds latency on CPU (a cross-encoder forward pass per candidate, reported as the `rerank` stage time) but no LLM tokens. If the service is down while reranking is enabled, the optional stage is skipped and the request is marked degraded. That TEI 1.9.4 serves this model on CPU was checked against the model's architecture only; the first run of the `rerank` profile has to confirm it.

- **Amendment (hosted reranker option):** the local `tei-rerank` service is heavy on a small machine (a 2.2 GB model and a CPU cross-encoder), so a hosted rerank API can be used instead. `RAGQA_RERANKER_KIND=api` selects `ApiReranker`, which sends the Cohere-style request (`model`, `query`, `documents`, `top_n`) to `<RAGQA_RERANKER_URL>/rerank` and reads `results[].index` and `relevance_score`. OpenRouter serves this at `https://openrouter.ai/api/v1/rerank` (for example `cohere/rerank-v3.5` at $0.001 per search, as listed on its model page on 2026-11-05), and the LiteLLM proxy exposes the same shape at `/v1/rerank`. With OpenRouter the key (`RAGQA_RERANKER_API_KEY`) is in the app's environment, an exception to ADR-0016, so pointing the URL at the proxy is preferred when the proxy supports the model (not checked for OpenRouter models). The reranker that scored a request is recorded in the pipeline state (`reranker`, for example `api:cohere/rerank-v3.5`) and in `finished.json` of a run, and counts as a pinned model: a run that reranks with a hosted model measures that model, not `bge-reranker-v2-m3`.
