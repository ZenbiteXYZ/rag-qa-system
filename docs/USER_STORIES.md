# RAG QA System: idea, user stories

Status: draft v0.1

## 1. Idea

A service where a user uploads documents (PDF, DOCX, Markdown, TXT, HTML), the system parses and indexes them, keeps every version of each document, and answers questions grounded in the relevant documents with citations.

### Goals
- Working end-to-end QA over user documents with source citations.
- Document versioning: upload a new version, query a specific version or the latest, see what changed.
- Pluggable chunking and retrieval strategies, selectable per query and per collection.
- Evaluation: compare strategies on a golden question set with retrieval and answer metrics.

### Non-goals (v1)
- Multi-tenant auth, billing, sharing between users.
- Real-time collaborative editing.
- OCR-heavy scans and complex table understanding (later stretch).

## 2. Personas

- **Reader**: uploads documents and asks questions. Wants correct answers with proof.
- **Experimenter** (the author): swaps chunkers and retrieval techniques, runs evaluations, inspects intermediate steps.

## 3. User stories

Priority: M = must (MVP), S = should, C = could.

### Epic A: Document management
| ID | Story | Acceptance criteria | P |
|----|-------|---------------------|---|
| A1 | As a Reader, I upload a document so that I can ask questions about it. | `.md` and `.docx` accepted, everything else (including PDF) rejected with a clear error. Upload returns a document id and an ingestion status. | M |
| A2 | As a Reader, I see the ingestion status of a document. | States: queued, parsing, chunking, embedding, ready, failed (with reason). | M |
| A3 | As a Reader, I list and delete my documents. | Deleting removes the document, its versions and its vectors. | M |
| A4 | As a Reader, I attach metadata (title, tags, author, date) to a document. | Metadata is stored and filterable at query time. | S |
| A5 | As a Reader, I upload a batch of files at once. | Each file gets its own status, one failure does not block the others. | C |

### Epic B: Versioning
| ID | Story | Acceptance criteria | P |
|----|-------|---------------------|---|
| B1 | As a Reader, I upload a new version of an existing document. | Version number increments, the previous version is kept. Identical content (same hash) is detected and not re-indexed. | M |
| B2 | As a Reader, I see the version history of a document. | List with version number, upload time, content hash, and size. | M |
| B3 | As a Reader, I ask a question against the latest version (default) or a chosen version. | Retrieval is filtered by version. Citations show the version. | M |
| B4 | As a Reader, I see what changed between two versions. | Text diff of the parsed content (added, removed, changed). | C |
| B5 | As a Reader, I roll back to an earlier version as "latest". | Latest pointer moves, no re-embedding needed. | C |

### Epic C: Question answering
| ID | Story | Acceptance criteria | P |
|----|-------|---------------------|---|
| C1 | As a Reader, I ask a question and get an answer based on my documents. | Answer is generated only from retrieved context. If context is insufficient, the system says so instead of guessing. | M |
| C2 | As a Reader, I see citations for each answer. | Each claim links to document, version, page or section, and the quoted chunk. | M |
| C3 | As a Reader, I restrict a question to selected documents, tags or metadata. | Filter applied before vector search. | M |
| C4 | As a Reader, I ask follow-up questions in a conversation. | History is used to rewrite the follow-up into a standalone query. | S |
| C5 | As a Reader, I get streamed answers. | Tokens arrive incrementally. | C |
| C6 | As a Reader, I give thumbs up or down on an answer. | Feedback stored with the full query trace. | C |

### Epic D: Retrieval techniques
| ID | Story | Acceptance criteria | P |
|----|-------|---------------------|---|
| D1 | As an Experimenter, I choose a chunking strategy per collection. | At least: fixed-size with overlap, recursive/structure-aware, semantic. Later: parent-child, sentence-window. Strategy and params are stored with the index. | M |
| D2 | As an Experimenter, I enable Query Rewriting. | Rewritten query is visible in the trace. Can be toggled per request. | M |
| D3 | As an Experimenter, I enable HyDE. | Hypothetical answer is generated, embedded, and used for retrieval. Visible in the trace. | M |
| D4 | As an Experimenter, I enable Self-Query. | The question is parsed into semantic query plus metadata filters (document, version, date, tags). Parsed filters are visible and validated against the metadata schema. Filter values are checked against known stored values; an unknown value is fuzzy-matched or returned to the user for clarification (ADR-0008). `version` and `is_latest` are system filters, not set by Self-Query. | M |
| D5 | As an Experimenter, I enable hybrid search (dense plus BM25) and reranking. | Fusion method and reranker are configurable. | S |
| D6 | As an Experimenter, I enable multi-query and decomposition for complex questions. | Sub-queries and merged results visible in the trace. | C |
| D7 | As an Experimenter, I inspect a full trace of a request. | Trace shows each stage: rewritten query, filters, retrieved chunks with scores, reranked order, final prompt, answer, latency and token cost per stage. | M |

### Epic E: Evaluation
| ID | Story | Acceptance criteria | P |
|----|-------|---------------------|---|
| E1 | As an Experimenter, I maintain a golden set of questions with expected source chunks and reference answers. | Stored in a file, versioned in git. | M |
| E2 | As an Experimenter, I run the golden set against a pipeline config and get retrieval and answer metrics. | Report includes the baseline (plain dense retrieval), metric definitions, and results across several runs, so a difference is not mistaken for noise. | M |
| E3 | As an Experimenter, I compare configs side by side. | Table of configs against metrics, plus per-question diffs. | S |

### Epic F: Operations
| ID | Story | Acceptance criteria | P |
|----|-------|---------------------|---|
| F1 | As the author, ingestion runs asynchronously. | Upload returns immediately, a worker processes the document. | M |
| F2 | As the author, configs live in files and runs are reproducible. | Seeds, model names, and library versions are logged per run. | M |
| F3 | As the author, I run everything with one command. | Docker Compose brings up API, worker, Qdrant and Postgres. | M |

User flows are in [USER_FLOWS.md](USER_FLOWS.md). Architecture decisions are in [ADR.md](ADR.md).
