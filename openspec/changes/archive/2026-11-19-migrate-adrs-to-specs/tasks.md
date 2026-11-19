# Tasks

## 1. Author capability deltas

- [x] 1.1 Write `specs/document-model/spec.md` (ADR-0001, ADR-0021); verify it has a Purpose and requirements with WHEN/THEN scenarios
- [x] 1.2 Write `specs/document-ingestion/spec.md` (ADR-0003, ADR-0011, ADR-0014, ADR-0017); verify structure as above
- [x] 1.3 Write `specs/vector-indexes/spec.md` (ADR-0002, ADR-0013, ADR-0018); verify structure as above
- [x] 1.4 Write `specs/query-pipeline/spec.md` (ADR-0007, ADR-0008, ADR-0010, ADR-0022, ADR-0023); verify structure as above
- [x] 1.5 Write `specs/model-serving/spec.md` (ADR-0009, ADR-0012, ADR-0016, ADR-0023 amendment); verify structure as above
- [x] 1.6 Write `specs/evaluation/spec.md` (ADR-0005, ADR-0006, ADR-0019) including the frozen-protocol governance requirement; verify structure as above
- [x] 1.7 Write `specs/observability/spec.md` (ADR-0015, ADR-0020) with the supersession note; verify structure as above
- [x] 1.8 Write `specs/api/spec.md` (ADR-0004); verify structure as above

## 2. Verify

- [x] 2.1 Run `openspec validate migrate-adrs-to-specs --strict` and confirm it passes (every requirement has scenarios, valid delta headers, descriptions within length)
- [x] 2.2 Coverage check: confirm all 23 ADRs (0001 through 0023) appear in requirement provenance across the deltas, with none dropped

## Workflow follow-up

- Archive the change after verification passes, seeding `openspec/specs/` through the delta-to-spec path.
- Verify the archived result with `openspec list --specs` (expect the 8 capabilities).
