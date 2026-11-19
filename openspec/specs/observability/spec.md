# observability Specification

## Purpose
Defines structured logging, MLflow request traces and evaluation runs, Prometheus metrics, and the Compose profiles that gate the optional observability and model services.

## Requirements

### Requirement: Structured JSON logs
Every log line SHALL render as JSON, including stdlib loggers, with request_id, job_id and trace_id from context and secret redaction. Container logs SHALL ship to Loki through Grafana Alloy. (ADR-0020)

#### Scenario: Log line carries context and redacts secrets
- **WHEN** a request or job logs a line
- **THEN** it is JSON with the bound context ids and secrets redacted

### Requirement: MLflow traces and runs
The system SHALL store request traces (one span per pipeline stage) and evaluation runs (config as params, metrics, report and golden set version as artifacts) in MLflow, using a separate Postgres database and an artifact volume. The trace id returned by POST /query is the MLflow trace id. (ADR-0015)

#### Scenario: Query returns the MLflow trace id
- **WHEN** POST /query completes
- **THEN** the trace id it returns is the MLflow trace id

### Requirement: Unified tracing
One OpenTelemetry tracer provider SHALL be shared by MLflow and the FastAPI and httpx auto-instrumentation, so infrastructure and pipeline spans land in one MLflow trace. The worker links its job trace to the API trace through W3C headers on the RabbitMQ message; there is no OTLP exporter. (ADR-0020)

#### Scenario: Infrastructure and pipeline spans in one trace
- **WHEN** a request is traced
- **THEN** infrastructure spans and pipeline spans land in one MLflow trace

#### Scenario: Worker trace linked to the API trace
- **WHEN** the worker processes a job
- **THEN** its job trace is linked to the API trace through W3C headers on the message

### Requirement: Prometheus metrics
The API SHALL expose prometheus-client metrics at /metrics and the worker on port 9100; Prometheus also scrapes Qdrant, RabbitMQ and TEI, and Grafana has provisioned datasources and one operations dashboard. This supersedes the ADR-0015 deferral of Prometheus and Grafana. (ADR-0020, ADR-0015)

#### Scenario: API and worker expose metrics
- **WHEN** Prometheus scrapes the application
- **THEN** the API serves /metrics and the worker serves metrics on port 9100

### Requirement: Compose profiles
The default Compose set SHALL be Postgres, Qdrant, RabbitMQ, MLflow, API and worker; the ml profile adds TEI and LiteLLM, and the obs profile adds Prometheus, Loki, Alloy and Grafana. (ADR-0020)

#### Scenario: Obs profile adds observability services
- **WHEN** Compose runs with the obs profile
- **THEN** Prometheus, Loki, Alloy and Grafana are added to the default set

### Requirement: Tracing opt-in
Tracing SHALL be off unless RAGQA_TRACING_ENABLED is set. (ADR-0020)

#### Scenario: Tracing disabled by default
- **WHEN** RAGQA_TRACING_ENABLED is not set
- **THEN** tracing is off
