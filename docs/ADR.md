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
- **Decision:** The author writes about 30 questions by hand, some with metadata cues ("in the latest guide", "by author X"). Synthetic questions are a supplement only. The set is stored in a file under version control.
- **Consequences:** Small set, so differences between configs must be reported with variance across runs and not as point estimates. Writing the set takes real time.

## ADR-0006: Evaluation corpus: Kubernetes docs

- **Status:** Proposed
- **Context:** The corpus needs Markdown, real version differences, and usable metadata. The author delegated the choice.
- **Decision:** `kubernetes/website`, `content/en/docs`, license CC BY 4.0. Take 2 or 3 `release-1.xx` branches (branches 1.30 to 1.36 exist at the time of writing). Use a subset such as `concepts/` and `tasks/`, English only. Front matter provides `title`, `content_type` (concept, task, tutorial, reference), `reviewers` and `weight`.
- **Consequences:** Attribution is required. Whether `reviewers` can serve as `author` is unverified (see ADR-0007). To be confirmed before building the golden set.

## ADR-0007: Metadata schema

- **Status:** Proposed
- **Context:** Self-Query is only as useful as the metadata it can filter on, and the fields must be ones users mention in questions. Version fields are rarely mentioned and a wrong guess silently returns stale answers.
- **Decision:**
  - Document level: `doc_type`, `tags`, `author`, `doc_date` (date stated by the content), `uploaded_at`, `language`.
  - Chunk level: `section_path` (heading hierarchy), position.
  - System filters, not available to Self-Query: `version`, `is_latest`.
- **Consequences:** The `author` field may be sparse in the chosen corpus (`reviewers` is missing in many files). Fallbacks: derive an author from git history per file, or use `reviewers` and rename the field.

## ADR-0008: Self-Query filter value validation

- **Status:** Accepted
- **Context:** The LLM may invent a filter value that does not exist, for example `author="Ivan"` when the stored value is `"I. Petrov"`. Applying it silently returns empty or wrong results.
- **Decision:** Filter values are checked against the known values stored in Postgres. For an unknown value, fuzzy matching finds candidates. If one candidate is clearly best, it is used and the substitution is reported in the trace. Otherwise the system returns the nearest candidates and asks the user to choose.
- **Consequences:** Needs a table of known values per filterable field, kept in sync on ingestion and deletion. The "clearly best" threshold is to be calibrated on real corpus values. With a stateless pipeline, the clarification is a "needs clarification" response listing candidates, and the client resends the request with the chosen value.

## ADR-0009: Models are configurable: local embeddings, API LLM

- **Status:** Accepted
- **Context:** Embeddings are called on every chunk and every query, so local inference saves cost. LLM quality matters for rewriting, HyDE, Self-Query parsing and generation.
- **Decision:** Embedding model and LLM are selected in config files. Embeddings run locally, the LLM is called through an API. Model names and versions are logged per run.
- **Consequences:** Changing the embedding model requires re-indexing (vector dimensions and semantics change), so the model name is stored with the index. Specific models are still to be chosen.

## ADR-0010: Pipeline shape: fixed sequence of stages

- **Status:** Accepted
- **Context:** An agent that chooses its own tool calls is flexible, but a metric change could then come from a different tool path and not from the technique being tested.
- **Decision:** The query pipeline is a fixed sequence of stages, and each technique is switched by config. Wrapping the pipeline as an agent tool is a later experiment.
- **Consequences:** Effects of single techniques can be attributed, and runs are more repeatable. Less adaptive behavior on complex questions until multi-query or decomposition (story D6) is added as another stage.
