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
    mlflow_tracking_uri: str = "http://localhost:5000"
    files_dir: Path = Path("data/files")

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


@lru_cache
def get_settings() -> Settings:
    # Required fields are filled from the environment, which mypy cannot see.
    return Settings()  # type: ignore[call-arg]
