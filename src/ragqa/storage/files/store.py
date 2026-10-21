"""Raw uploaded files on the shared volume (ADR-0017).

Layout: `<root>/<document_id>/<version_id>/<safe file name>`. Writes are atomic: a file
is written under a temporary name and renamed, so a reader never sees half a file.
"""

import asyncio
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(filename: str) -> str:
    """The base name with unusual characters replaced. Keeps the extension."""
    name = _UNSAFE.sub("_", Path(filename).name).strip("._") or "file"
    return name[-200:]


class FileStore:
    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def path_for(self, document_id: uuid.UUID, version_id: uuid.UUID, filename: str) -> str:
        """Relative path, stored in `versions.raw_path`."""
        return f"{document_id}/{version_id}/{safe_filename(filename)}"

    def _resolve(self, relative: str) -> Path:
        path = (self._root / relative).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError(f"{relative!r} points outside the file store")
        return path

    async def save(self, relative: str, data: bytes) -> None:
        await asyncio.to_thread(self._write, self._resolve(relative), data)

    @staticmethod
    def _write(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    async def read(self, relative: str) -> bytes:
        return await asyncio.to_thread(self._resolve(relative).read_bytes)

    async def delete(self, relative: str) -> None:
        await asyncio.to_thread(self._resolve(relative).unlink, missing_ok=True)

    async def delete_document(self, document_id: uuid.UUID) -> None:
        """Every version's files. Safe to call again."""
        await asyncio.to_thread(shutil.rmtree, self._resolve(str(document_id)), ignore_errors=True)
