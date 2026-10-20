"""Error classes that tell the worker whether a failed job is worth retrying (ADR-0011)."""


class IngestionError(Exception):
    """Base class for ingestion failures."""


class PermanentIngestionError(IngestionError):
    """Retrying cannot help, for example an unparseable or empty file. The job fails at once."""


class TransientIngestionError(IngestionError):
    """A dependency was unavailable (embedder, Qdrant, Postgres). The job is retried.

    Any exception that is not a `PermanentIngestionError` is also treated as transient, so
    raising this is only needed to make the intent explicit.
    """


class UnsupportedFormatError(PermanentIngestionError):
    """No parser is registered for the file's extension."""
