# Spec Delta

## Purpose

Defines how embedding and LLM models are selected, served and accessed: a shared local embedder used by API and worker, and LLM calls through a LiteLLM proxy.

## ADDED Requirements

### Requirement: Configurable models
Embedding model and LLM SHALL be selected in config; embeddings run locally and the LLM through an API. Model names and versions are logged per run, the embedding model name is stored with the index, and changing it requires re-indexing. (ADR-0009)

#### Scenario: Model name stored with the index
- **WHEN** an index is created
- **THEN** its embedding model name is stored with it

#### Scenario: Changing the embedder creates a new index
- **WHEN** the embedding model changes
- **THEN** a new index is created rather than modifying an existing one

### Requirement: Shared embedder service
Both API and worker SHALL call one embedding service behind an Embedder interface, with an in-process implementation of the same interface for development and tests, so index and query use one model by construction. (ADR-0012)

#### Scenario: API and worker use the same embedder
- **WHEN** the API embeds a query and the worker embeds chunks
- **THEN** both call the same embedding service and model

#### Scenario: In-process implementation for tests
- **WHEN** running development or tests
- **THEN** an in-process implementation of the Embedder interface can be used

### Requirement: Query and passage prefixes
The Embedder interface SHALL distinguish query and passage inputs. For the base model, queries get the prefix `query: ` and documents none, added by the client because the server does not apply it. (ADR-0012)

#### Scenario: Query prefixed, passage not
- **WHEN** the base model embeds a query and a passage
- **THEN** the client adds `query: ` to the query and no prefix to the passage

### Requirement: LLM access through a proxy
The app SHALL call the LLM through a LiteLLM proxy over an OpenAI-compatible API; pipeline model names are aliases defined in the proxy config, and provider keys live only in the proxy environment, not the app. (ADR-0016)

#### Scenario: Stage model switched by alias
- **WHEN** a stage is pointed at another model or provider
- **THEN** it is a proxy-config alias change, with one client implementation in the app

#### Scenario: App holds no provider keys
- **WHEN** the app calls the LLM
- **THEN** provider keys are held only in the proxy environment

### Requirement: Reranker key exception
Provider keys SHALL stay out of the app, except a hosted reranker pointed directly at OpenRouter uses an app-environment key; pointing the reranker at the proxy is preferred when it supports the model. (ADR-0023 amendment, ADR-0016)

#### Scenario: OpenRouter reranker uses an app-env key
- **WHEN** the hosted reranker URL points directly at OpenRouter
- **THEN** its key is in the app environment as a documented exception to the proxy-only rule
