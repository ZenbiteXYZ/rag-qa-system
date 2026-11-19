# document-ingestion Specification

## Purpose
Covers uploading and versioning documents, transporting ingestion jobs to a worker, storing raw files, and keeping Postgres and Qdrant consistent through ordered steps with compensation.

## Requirements

### Requirement: Version per upload
A version SHALL be one uploaded file; a new version is re-parsed and fully re-embedded, and identical content (the same hash) SHALL create no new version. (ADR-0003)

#### Scenario: New file creates a new version
- **WHEN** a user uploads a changed file for an existing document
- **THEN** a new version is created, re-parsed and fully re-embedded

#### Scenario: Identical content creates no version
- **WHEN** a user uploads a file whose content hash matches the current version
- **THEN** no new version is created

### Requirement: Job transport over RabbitMQ
The system SHALL carry ingestion jobs from the API to a separate worker over RabbitMQ, while job status (queued, parsing, chunking, embedding, ready, failed, dead) stays in Postgres. (ADR-0011)

#### Scenario: API publishes and worker consumes
- **WHEN** the API commits an upload
- **THEN** it publishes an ingestion job that the worker consumes

#### Scenario: Status lives in Postgres
- **WHEN** a job moves through its lifecycle
- **THEN** its status is recorded in Postgres, not in the queue message

### Requirement: Retry and dead-letter
The system SHALL retry a failed job with backoff up to 3 times and then dead-letter it; permanent errors (such as an unparseable file) SHALL skip retries and fail at once. (ADR-0011)

#### Scenario: Transient failure retried then dead-lettered
- **WHEN** a job fails with a retryable error
- **THEN** it is retried with backoff up to 3 times and then sent to the dead-letter queue

#### Scenario: Permanent error fails immediately
- **WHEN** a job fails with a permanent error
- **THEN** it fails at once without retries

### Requirement: At-least-once idempotency
The system SHALL make every ingestion step repeatable because delivery is at-least-once, and a maintenance loop SHALL republish jobs stuck in queued past a timeout. (ADR-0011, ADR-0014)

#### Scenario: Redelivered job does not duplicate work
- **WHEN** a job message is delivered more than once
- **THEN** the repeated steps overwrite or no-op rather than duplicate data

#### Scenario: Stuck queued job republished
- **WHEN** a job stays queued past the timeout
- **THEN** the maintenance loop republishes it

### Requirement: Raw files on a shared volume
The system SHALL store raw uploaded files on a Docker volume mounted into both API and worker, accessed through a swappable file-storage module. (ADR-0017)

#### Scenario: Worker reads the file the API saved
- **WHEN** the worker ingests an upload
- **THEN** it reads the raw file from the shared volume the API wrote to

### Requirement: Ordered ingestion with compensation
The system SHALL ingest a version in order: write Qdrant points with is_latest=false, commit one Postgres transaction (version ready, latest pointer moved, job finished), then set is_latest=true on the new points and false on the old. A brief overlap is preferred over a gap. (ADR-0014)

#### Scenario: Default queries hide unfinished work
- **WHEN** points are written before the Postgres commit
- **THEN** they carry is_latest=false and default queries cannot see them

#### Scenario: Failure before the commit cleans up
- **WHEN** writing points or the Postgres transaction fails
- **THEN** the version's points are deleted by version_id and the job is marked failed or retried idempotently

#### Scenario: Failure during the flip is retried
- **WHEN** the is_latest flip step fails
- **THEN** it is retried and the reconciler repairs any mismatch

### Requirement: Latest version source of truth
The system SHALL treat Postgres documents.latest_version_id as the source of truth for which version is latest; the Qdrant is_latest payload is a denormalized copy. A reconciler SHALL repair mismatches and check exactly one latest version per document in both stores. (ADR-0014)

#### Scenario: Reconciler repairs a mismatch
- **WHEN** the Qdrant is_latest copy disagrees with Postgres
- **THEN** the reconciler repairs it to match the Postgres latest pointer

### Requirement: Deletion order and tombstone
The system SHALL delete a document by marking it deleted in Postgres first (hidden), then deleting vectors by filter, then raw files, all idempotent, and keeping a tombstone row. (ADR-0014)

#### Scenario: Deletion hides then removes
- **WHEN** a document is deleted
- **THEN** it is hidden in Postgres first, then its vectors and raw files are removed idempotently

#### Scenario: Tombstone retained
- **WHEN** deletion completes
- **THEN** a tombstone row remains so the deletion can be reported
