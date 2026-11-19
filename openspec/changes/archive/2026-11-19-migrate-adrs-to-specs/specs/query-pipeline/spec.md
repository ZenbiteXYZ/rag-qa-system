# Spec Delta

## Purpose

Defines the fixed query stage sequence, Self-Query filter validation against known values, the metadata schema, reranking, and the protocol-v2 agent boundary.

## ADDED Requirements

### Requirement: Fixed stage sequence
The query pipeline SHALL run a fixed sequence of stages (rewrite, Self-Query, HyDE, retrieve, rerank, context, generate), each technique switched by config. Each stage writes only its declared fields, and an optional stage that fails degrades the request. (ADR-0010)

#### Scenario: Technique toggled by config
- **WHEN** a request config enables or disables a technique
- **THEN** the corresponding stage runs or is skipped without changing the sequence

#### Scenario: Optional stage failure degrades the request
- **WHEN** an optional stage fails
- **THEN** the request continues and is marked degraded

### Requirement: Metadata schema
The system SHALL expose document-level filters doc_type, tags, author, doc_date, uploaded_at and language, and chunk-level section_path and position. version and is_latest are system filters, not available to Self-Query. (ADR-0007, marked Proposed but implemented)

#### Scenario: Self-Query cannot use system filters
- **WHEN** Self-Query extracts filters
- **THEN** it can use the document and chunk metadata fields but not version or is_latest

#### Scenario: Absent author is synthetic
- **WHEN** the corpus has no reliable author for a document
- **THEN** author metadata is assigned from a versioned mapping file

### Requirement: Filter value validation
Self-Query filter values SHALL be checked against known values in Postgres. For an unknown value, fuzzy matching finds candidates; if one is clearly best it is used and the substitution reported in the trace, otherwise the system returns nearest candidates as a needs-clarification response. (ADR-0008)

#### Scenario: Unknown value substituted and reported
- **WHEN** an extracted value is unknown but one fuzzy candidate is clearly best
- **THEN** that candidate is used and the substitution is reported in the trace

#### Scenario: Ambiguous value returns candidates
- **WHEN** no candidate is clearly best
- **THEN** the system returns a needs-clarification response listing the nearest candidates

#### Scenario: Client resends with the chosen value
- **WHEN** the client resends the request with a chosen candidate
- **THEN** the pipeline runs with that value

### Requirement: Reranking kinds
The rerank stage SHALL reorder candidates with a cross-encoder served by TEI or a hosted rerank API, selected by config, and SHALL record which reranker scored the request as a pinned model. (ADR-0023)

#### Scenario: TEI reranker reorders candidates
- **WHEN** reranking is enabled with the TEI kind
- **THEN** a cross-encoder reorders the candidates

#### Scenario: Hosted reranker recorded
- **WHEN** reranking is enabled with the API kind
- **THEN** the hosted reranker that scored the request is recorded in the pipeline state and the run record

### Requirement: Reranker availability
A request that enables reranking on a deployment without a reranker URL SHALL be rejected with 422; if the reranker is down while enabled, the optional stage is skipped and the request marked degraded. (ADR-0023)

#### Scenario: No reranker URL rejects the request
- **WHEN** a request enables reranking but no reranker URL is configured
- **THEN** the request is rejected with 422

#### Scenario: Reranker outage degrades the request
- **WHEN** the reranker is down while reranking is enabled
- **THEN** the stage is skipped and the request is marked degraded

### Requirement: Agent mode deferred to protocol v2
Protocol v1 SHALL run on the fixed pipeline only. Agent mode is added under protocol v2 as a second mode, with the pipeline staying the baseline and control; results from the two modes are never mixed in one protocol version. (ADR-0022, ADR-0010)

#### Scenario: v1 request uses the fixed pipeline
- **WHEN** a protocol v1 request is answered
- **THEN** it runs the fixed pipeline with no agent tool calls
