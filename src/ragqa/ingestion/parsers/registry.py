"""Parser registry: file extension -> parser (ADR-0021).

Adding a format means writing a `Parser` and registering it. The domain types and the API
do not change: the API asks the registry which extensions it accepts.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

from ragqa.domain.documents import ParsedDocument
from ragqa.ingestion.errors import UnsupportedFormatError


class Parser(Protocol):
    format_key: str  # stored as `ParsedDocument.source_format`, lowercase letters and digits
    extensions: tuple[str, ...]  # lowercase, with the dot

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        """Raise `PermanentIngestionError` for input that retrying cannot fix."""
        ...


class ParserRegistry:
    def __init__(self, parsers: Iterable[Parser] = ()) -> None:
        self._by_extension: dict[str, Parser] = {}
        for parser in parsers:
            self.register(parser)

    def register(self, parser: Parser) -> None:
        for extension in parser.extensions:
            if extension in self._by_extension:
                raise ValueError(
                    f"{extension} is already handled by {self._by_extension[extension].format_key}"
                )
            self._by_extension[extension] = parser

    def for_filename(self, filename: str) -> Parser:
        extension = Path(filename).suffix.lower()
        try:
            return self._by_extension[extension]
        except KeyError:
            what = extension or "files without an extension"
            supported = ", ".join(self.extensions)
            raise UnsupportedFormatError(f"no parser for {what}; supported: {supported}") from None

    @property
    def extensions(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_extension))
