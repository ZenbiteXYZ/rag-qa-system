from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogFormat = Literal["json", "console"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


class Settings(BaseSettings):
    """Service configuration. All values come from RAGQA_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="RAGQA_")

    # No defaults for URLs that carry credentials.
    postgres_dsn: str
    rabbitmq_url: str

    qdrant_url: str = "http://localhost:6333"
    embedder_url: str = "http://localhost:8080"
    litellm_url: str = "http://localhost:4000"
    litellm_api_key: SecretStr | None = None
    # LiteLLM alias used by every LLM stage unless a request overrides it (ADR-0019).
    default_model: str = "haiku-5.5"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2  # the openai client retries 408, 409, 429 and 5xx by itself
    mlflow_tracking_uri: str = "http://localhost:5000"
    files_dir: Path = Path("data/files")

    # Reranker (ADR-0023). Unset means `rerank.enabled` is rejected on this deployment.
    reranker_url: str | None = None
    # `POST /query` may return the whole pipeline state (stage outputs, prompts, chunk texts)
    # when asked. The evaluation runner needs it; leave it off elsewhere.
    allow_query_debug: bool = False

    # Documents
    max_upload_bytes: int = 20 * 1024 * 1024
    default_index: str | None = None  # used when an upload names none and several exist

    # Embedder (ADR-0012). The base model wants this prefix on queries and none on passages.
    # `embedder_model` is the one dense model this deployment serves, and `POST /indexes`
    # rejects any other.
    embedder_model: str = "Snowflake/snowflake-arctic-embed-m-v2.0"
    # Pinned in ADR-0019; the token counter loads the tokenizer at this revision.
    embedder_revision: str = "95c2741480856aa9666782eb4afe11959938017f"
    embedder_timeout_s: float = 120.0
    embedder_query_prefix: str = "query: "
    embedder_passage_prefix: str = ""
    embedder_batch_size: int = 32

    # Observability
    service_name: str = "ragqa-api"
    env: str = "dev"
    log_level: LogLevel = "INFO"
    log_format: LogFormat = "json"
    # Off by default so tests and local runs without an MLflow server stay quiet.
    tracing_enabled: bool = False
    mlflow_experiment: str = "ragqa-traces"
    worker_metrics_port: int = 9100

    # Ingestion queue (ADR-0011)
    ingest_max_retries: int = 3
    ingest_retry_base_delay_s: float = 5.0
    # Below Compose's stop_grace_period, so an in-flight job can finish before SIGKILL.
    worker_shutdown_grace_s: float = 25.0
    # Maintenance loop in the worker: re-send stuck `queued` jobs and finish deletions.
    maintenance_interval_s: float = 30.0
    republish_after_s: float = 60.0


@lru_cache
def get_settings() -> Settings:
    # Required fields are filled from the environment, which mypy cannot see.
    return Settings()  # type: ignore[call-arg]
