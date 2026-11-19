# api Specification

## Purpose
Defines the API-only interface contract: a FastAPI service with no UI, where the auto-generated docs page is the test interface and the schemas are the client contract.

## Requirements

### Requirement: API-only service
The system SHALL expose a FastAPI service only, with no UI; the auto-generated /docs page is the test interface. (ADR-0004)

#### Scenario: Docs page is the test interface
- **WHEN** a developer wants to exercise the service
- **THEN** the auto-generated /docs page is available as the test interface

### Requirement: Published API contract
The API SHALL publish request and response schemas that clients and any later UI depend on, so the schemas are treated as a contract from the start. (ADR-0004)

#### Scenario: Schemas published for clients
- **WHEN** a client integrates with the service
- **THEN** it depends on the published request and response schemas
