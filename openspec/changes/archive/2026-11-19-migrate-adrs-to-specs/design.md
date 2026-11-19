# Design

## Context

See proposal.md - Why. The system is already built; `docs/ADR.md` holds 23 decision records and OpenSpec has no specs. ADR ids are cited in 60+ places in `src/`, plus `CLAUDE.md`, `ARCHITECTURE.md` and `ML_DESIGN.md`, so the migration must keep them traceable. This is a brownfield bootstrap: we seed the spec inventory through one change archived via the normal delta-to-spec path, not a behavior change.

## Goals / Non-Goals

**Goals:**
- Translate each accepted ADR decision into testable requirements (SHALL + scenarios) with ADR provenance.
- Preserve the current truth where ADRs evolved (supersession, amendment) and record boundaries (agent deferral, protocol freeze).
- Keep `docs/ADR.md` as the decision history; touch no code.

**Non-Goals:**
- Absorbing `ARCHITECTURE.md` or `ML_DESIGN.md` into specs.
- Editing ADR statuses or content (including ADR-0007's Proposed status).
- Any change to `src/`, tests, the API contract, dependencies, or the protected core.

## Decisions

### Translation rule: behavior, not implementation
Specs state observable behavior and external constraints. External system names that define an interface (Qdrant, Postgres, RabbitMQ, TEI, LiteLLM, MLflow, Prometheus, Grafana, Loki, Alloy, FastAPI) and pinned model identifiers (normative protocol values) are retained. Internal class/function names and library choices (for example `TeiReranker`, `aio-pika`, frozen Pydantic models) are generalized to behavior. Rationale: a spec should survive an implementation swap; the technology choices that are themselves the decision (ADR-0011, ADR-0012, ADR-0016) stay as interface constraints. Alternative considered: strip all technology names, rejected because it would erase the essence of several ADRs.

### Provenance convention
Each requirement's description ends with its source ADR(s), e.g. `(ADR-0013, ADR-0018)`. This keeps every requirement traceable without touching the existing code references. Descriptions are kept under 500 characters; detail and edge cases live in scenarios.

### Capability map (all 23 ADRs covered)
| Capability | ADRs |
|---|---|
| document-model | 0001, 0021 |
| document-ingestion | 0003, 0011, 0014, 0017 |
| vector-indexes | 0002, 0013, 0018 |
| query-pipeline | 0007, 0008, 0010, 0022, 0023 |
| model-serving | 0009, 0012, 0016, 0023 (amendment) |
| evaluation | 0005, 0006, 0019 |
| observability | 0015, 0020 |
| api | 0004 |

ADR-0002 is cross-cutting: its Qdrant side anchors `vector-indexes`, its Postgres-registry side is referenced by `document-ingestion`. ADR-0023 spans `query-pipeline` (rerank behavior) and `model-serving` (the key exception).

### Encoding the tricky cases
- **Supersession (0020 over 0015):** `observability` states the current full stack. The Prometheus requirement cites both ADRs and notes it supersedes the 0015 deferral. The deferral is not a requirement; history stays in ADR.md.
- **Proposed status (0007):** encoded as a normal requirement (the code implements it) with a provenance note flagging the discrepancy. ADR-0007's status is left for the owner to confirm; not silently edited.
- **Scheduling + sketch (0022):** not a behavioral requirement. A boundary requirement in `query-pipeline` records that v1 runs the fixed pipeline only and agent mode is a v2 addition; the sketch stays in ADR.md.
- **Amendment (0023):** the rerank requirements cover both TEI and API kinds, the 422 and degraded behaviors, and pinned-model recording; the key exception is a `model-serving` requirement.
- **Frozen protocol (0019):** `evaluation` reproduces the pinned values exactly and adds a governance requirement that any change creates protocol v2 with a new ADR and that versions are never mixed.

### Mechanics: bootstrap change then archive
One change with ADDED deltas for all 8 capabilities, archived to seed `openspec/specs/`. Alternative considered: author the main specs directly with no change. Rejected because the change gives an audit trail, a home for these translation rules, and a per-capability checklist, and it uses the sanctioned delta-to-spec path.

## Risks / Trade-offs

- [Specs drift from ADR.md over time] -> Provenance tags tie each requirement to its ADR; future behavior changes go through an OpenSpec change and new decisions through an ADR.
- [Technology names in specs reduce implementation-agnosticism] -> Deliberate trade-off for fidelity; only interface-defining names and pinned protocol values are kept (see translation rule).
- [A dense ADR loses nuance when split into requirements] -> Scenarios carry the edge cases (failure, compensation, rejection); design.md records the mapping so nothing is dropped silently.
- [ADR-0007 status ambiguity] -> Flagged in provenance and left for the owner; the requirement reflects implemented behavior.

## Migration Plan

1. Author proposal, the 8 delta specs, design and tasks (this change).
2. `openspec validate migrate-adrs-to-specs --strict`; confirm all 23 ADRs are mapped.
3. Archive the change to seed `openspec/specs/`; verify with `openspec list --specs`.
Rollback: the change is documentation only; removing the archived specs and the change directory restores the prior state. No code or data is affected.

## Open Questions

- Whether to add a one-line pointer at the top of `ADR.md` to `openspec/specs/`, and a short `CLAUDE.md` workflow note. Both are optional add-ons deferred to the owner; neither changes the specs or the task breakdown.
