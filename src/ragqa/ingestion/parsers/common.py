"""Small helpers shared by the parsers."""

from pathlib import Path
from typing import Any


def text_or_none(value: Any) -> str | None:
    """A stripped non-empty string, or None for anything else."""
    return (value.strip() or None) if isinstance(value, str) else None


def tags_from(value: Any) -> tuple[str, ...]:
    """Tags from a list, or from a comma separated string."""
    items = value.split(",") if isinstance(value, str) else value
    if not isinstance(items, list | tuple):
        return ()
    return tuple(t.strip() for t in items if isinstance(t, str) and t.strip())


def title_from_filename(filename: str) -> str:
    return Path(filename).stem.replace("-", " ").replace("_", " ").strip() or "Untitled"
