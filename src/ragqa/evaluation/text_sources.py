"""Text sources for the evidence validator (`ragqa.evaluation.evidence --text-source`).

    RAGQA_CORPUS_DIR=/path/to/corpus python -m ragqa.evaluation.evidence eval/golden \\
        --text-source ragqa.evaluation.text_sources:corpus_text

The corpus directory holds one folder per version with the docs below it, for example
`<corpus>/release-1.33/concepts/workloads/pods/_index.md`. The golden set's `document` is
the path below the version folder, `version` is the folder name.
"""

import os
from pathlib import Path

from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.ingestion.parsers import default_registry


def corpus_text(document: str, version: str) -> str:
    """Parser output for one corpus document, the text that quotes must match."""
    root = Path(os.environ["RAGQA_CORPUS_DIR"]).resolve()
    path = (root / version / document).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{version}/{document} points outside the corpus directory")
    data = path.read_bytes()
    try:
        return default_registry().for_filename(path.name).parse(data, filename=path.name).text
    except PermanentIngestionError as exc:
        # The validator reports ValueError as TEXT_UNAVAILABLE for this quote.
        raise ValueError(str(exc)) from exc
