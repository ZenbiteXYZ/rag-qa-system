"""Apply the registry migrations: `python -m ragqa.storage.postgres.migrate`.

Reads RAGQA_POSTGRES_DSN. Compose runs this once before the API and the worker start.
"""

import os
from pathlib import Path

from alembic import command
from alembic.config import Config

from ragqa.observability.logging import configure_logging, get_logger

log = get_logger(__name__)
MIGRATIONS = Path(__file__).parent / "migrations"


def alembic_config(dsn: str | None = None) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS))
    if dsn is not None:
        # Config values go through configparser interpolation, so `%` must be doubled.
        config.set_main_option("sqlalchemy.url", dsn.replace("%", "%%"))
    return config


def main() -> None:
    configure_logging(service="ragqa-migrate", env=os.environ.get("RAGQA_ENV", "dev"))
    command.upgrade(alembic_config(os.environ["RAGQA_POSTGRES_DSN"]), "head")
    log.info("migrations_applied")


if __name__ == "__main__":
    main()
