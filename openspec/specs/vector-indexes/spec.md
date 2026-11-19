# vector-indexes Specification

## Purpose
Defines the Qdrant and Postgres storage split and how chunks and vectors are laid out in one collection per index config, with dense and BM25 sparse vectors for hybrid retrieval.

## Requirements

### Requirement: Storage split
The system SHALL store chunks and vectors in Qdrant and the registry (documents, versions, the latest pointer, ingestion status and known metadata values) in Postgres. (ADR-0002)

#### Scenario: Vectors and registry separated
- **WHEN** a version is ingested
- **THEN** its chunks and vectors go to Qdrant and its registry rows go to Postgres

### Requirement: One collection per index config
The system SHALL create one Qdrant collection per index config, defined as (embedding model, chunking strategy and params). The user chooses the target index at upload and the index to search at query; the same version may be built in several indexes. (ADR-0013)

#### Scenario: Different params yield different collections
- **WHEN** two indexes use different chunking strategies or params
- **THEN** they are separate Qdrant collections

#### Scenario: One version built in several indexes
- **WHEN** a version is ingested into multiple indexes
- **THEN** each build is tracked with its own status

### Requirement: Index identity and keys
Each index SHALL have a surrogate id used by every foreign key and point id, with a unique name and collection; a rename SHALL keep point ids unchanged. (ADR-0013)

#### Scenario: Rename keeps point ids
- **WHEN** an index is renamed
- **THEN** its point ids are unchanged because they derive from the id, not the name

### Requirement: Deterministic point ids
The point id SHALL be a deterministic UUID v5 of (version id, index id, chunk index), so a retried upsert overwrites instead of duplicating. (ADR-0013)

#### Scenario: Retried upsert overwrites
- **WHEN** the same chunk is upserted again
- **THEN** it overwrites the existing point rather than creating a duplicate

### Requirement: Payload and payload indexes
Each point SHALL carry the defined payload fields (document_id, version, is_latest, doc_type, tags, author, doc_date, language, section_path, chunk_index, text) and a payload index on every filtered field. (ADR-0013)

#### Scenario: Filtered field has a payload index
- **WHEN** a field is used for filtering
- **THEN** a payload index exists for it

### Requirement: Dense and BM25 vectors at creation
Every collection SHALL define a named dense vector and a sparse bm25 vector with the IDF modifier at creation, and each chunk SHALL be upserted with both. (ADR-0018, ADR-0013)

#### Scenario: Collection created with both vectors
- **WHEN** a collection is created
- **THEN** it defines both the dense and the bm25 sparse vector

#### Scenario: Sparse cannot be added later
- **WHEN** a sparse vector is added to an existing dense-only collection
- **THEN** the operation is rejected, because both vectors must be defined at creation

### Requirement: Hybrid retrieval modes
The retrieve stage SHALL support dense, sparse and hybrid modes; hybrid is one query with a prefetch branch per vector fused by Reciprocal Rank Fusion. The evaluation baseline stays dense only, and the sparse branch never uses HyDE. (ADR-0018)

#### Scenario: Hybrid fuses dense and sparse
- **WHEN** a query runs in hybrid mode
- **THEN** one query runs a prefetch branch per vector and fuses them by RRF

#### Scenario: Sparse branch never uses HyDE
- **WHEN** the sparse branch builds its query text
- **THEN** it uses the Self-Query semantic query, else the rewritten query, else the question, and never HyDE
