# Proposal

## Why

The project's architectural decisions live only in `docs/ADR.md` as decision records (context, alternatives, consequences). OpenSpec is initialized but has no specs, so there is no machine-checkable behavioral contract for the system that is already built. Seeding capability specs from the accepted ADRs gives a living, validatable inventory of what the system must do, while keeping ADR.md as the rationale and the home for future decisions.

## What Changes

- Derive normative requirements (SHALL + scenarios) from the 23 ADRs and seed `openspec/specs/` through this bootstrap change, archived via the normal delta-to-spec path.
- Tag every requirement with its source ADR(s) so the 60+ existing ADR references in `src/`, `CLAUDE.md` and the docs stay traceable.
- Encode the current truth where ADRs evolved: ADR-0020 supersedes the ADR-0015 metrics deferral; the ADR-0023 amendment (hosted reranker) is included; ADR-0022's agent mode is recorded as a v2 boundary, not v1 behavior.
- Add an explicit governance requirement freezing evaluation protocol v1 (ADR-0019).
- Documentation only. No code, API, dependency or protected-core changes. `docs/ADR.md`, `ARCHITECTURE.md` and `ML_DESIGN.md` are left as they are.

## Capabilities

### New Capabilities

- `document-model`: supported formats, the normalized-text-with-spans document model, and the open parser registry (ADR-0001, ADR-0021).
- `document-ingestion`: per-upload versioning, RabbitMQ job transport and retries, raw-file storage, and Postgres/Qdrant consistency with compensation (ADR-0003, ADR-0011, ADR-0014, ADR-0017).
- `vector-indexes`: the Qdrant/Postgres storage split, one collection per index config, deterministic point ids, and dense plus BM25 hybrid vectors (ADR-0002, ADR-0013, ADR-0018).
- `query-pipeline`: the fixed stage sequence, Self-Query filter validation, the metadata schema, reranking, and the protocol-v2 agent boundary (ADR-0007, ADR-0008, ADR-0010, ADR-0022, ADR-0023).
- `model-serving`: configurable models, the shared TEI embedder, and LLM access through the LiteLLM proxy (ADR-0009, ADR-0012, ADR-0016).
- `evaluation`: the hand-written golden set, the Kubernetes corpus, and the frozen protocol v1 with its governance rule (ADR-0005, ADR-0006, ADR-0019).
- `observability`: structured logs, MLflow traces and runs, Prometheus metrics, and Compose profiles (ADR-0015, ADR-0020).
- `api`: the API-only interface contract (ADR-0004).

### Modified Capabilities

None. The spec inventory is empty; every capability above is introduced here.

## Impact

- Creates `openspec/specs/<capability>/spec.md` for the 8 capabilities on archive.
- Adds this change's artifacts under `openspec/changes/migrate-adrs-to-specs/` (proposal, design, tasks, delta specs); the archived change retains the migration rationale.
- No effect on `src/`, tests, the API contract, dependencies, or the protected core. ADR numbers remain valid everywhere they are cited.
