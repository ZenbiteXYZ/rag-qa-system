"""The ingest manifest: which registry ids hold which (document, release) of the corpus.

Retrieved chunks carry registry ids, evidence is tagged with corpus paths and releases. The
manifest, written by `load_corpus` and copied into every run, is the bridge between them.
"""

import json
import uuid
from pathlib import Path

from pydantic import BaseModel, ConfigDict


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ManifestVersion(_Model):
    document: str  # corpus path below the release folder
    release: str
    document_id: uuid.UUID
    version_id: uuid.UUID
    version: int  # the registry's version number of that document


class IngestManifest(_Model):
    index: str
    corpus_releases: tuple[str, ...]
    versions: tuple[ManifestVersion, ...]

    def by_version_id(self) -> dict[uuid.UUID, ManifestVersion]:
        return {v.version_id: v for v in self.versions}

    def version_number(self, document: str, release: str) -> int | None:
        for v in self.versions:
            if v.document == document and v.release == release:
                return v.version
        return None

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "IngestManifest":
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))
