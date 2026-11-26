"""PDF (.pdf) parser (ADR-0024).

Text-based PDFs only: an upload with no extractable text (a scanned or image-only
PDF) and a password-protected PDF are permanent errors. PDF has no heading
semantics, so sections are inferred best-effort from the font: a line whose
dominant span is clearly larger than the body size (or bold and larger), short,
and not monospaced is a heading, and the distinct heading sizes ranked
descending become levels 1..n. A document with no detectable heading is one flat
section. No code spans: PDF has no reliable code marker. Metadata comes from the
PDF Info dictionary.
"""

import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from typing import Any

import pymupdf
from pydantic import ValidationError

from ragqa.domain.documents import DocumentMetadata, ParsedDocument, normalize_text
from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.ingestion.parsers.builder import TextBuilder
from ragqa.ingestion.parsers.common import tags_from, text_or_none, title_from_filename

_BOLD = 1 << 4  # span flag bit: bold
_MONOSPACED = 1 << 3  # span flag bit: monospaced
_HEADING_SIZE_RATIO = 1.15  # a line this much larger than the body size is a heading
_MAX_HEADING_CHARS = 80  # a line longer than this is not a heading, however large

_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})")


def _doc_date(raw: Any) -> date | None:
    """`creationDate` is a PDF date string such as `D:20260301120000Z`."""
    if not isinstance(raw, str):
        return None
    digits = raw[2:] if raw.startswith("D:") else raw
    if match := _DATE.match(digits):
        try:
            return date(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:
            return None
    return None


@dataclass(frozen=True)
class _Line:
    """One visual line: its text, its dominant span's font size and flags."""

    text: str
    size: float
    flags: int

    def is_heading(self, body: float) -> bool:
        larger = self.size >= body * _HEADING_SIZE_RATIO or bool(
            self.flags & _BOLD and self.size > body
        )
        return larger and len(self.text) <= _MAX_HEADING_CHARS and not self.flags & _MONOSPACED


def _line_of(spans: list[dict[str, Any]]) -> _Line | None:
    if not spans:
        return None
    text = normalize_text("".join(str(span["text"]) for span in spans)).strip()
    if not text:
        return None
    dominant = max(spans, key=lambda span: len(str(span["text"])))
    return _Line(text=text, size=float(dominant["size"]), flags=int(dominant["flags"]))


def _blocks(doc: pymupdf.Document) -> list[list[_Line]]:
    """The text lines per text block, in reading order."""
    blocks: list[list[_Line]] = []
    for page in doc.pages():
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:  # 1 is an image
                continue
            lines = [
                line
                for raw_line in block["lines"]
                if (line := _line_of(raw_line["spans"])) is not None
            ]
            if lines:
                blocks.append(lines)
    return blocks


def _body_size(lines: list[_Line]) -> float:
    """The most common font size, weighted by how much text is set in it."""
    weights: Counter[float] = Counter()
    for line in lines:
        weights[round(line.size, 1)] += len(line.text)
    return max(weights, key=lambda size: weights[size])


def _levels(lines: list[_Line], body: float) -> dict[float, int]:
    """Heading sizes ranked descending into levels 1..n."""
    sizes = sorted({round(line.size, 1) for line in lines if line.is_heading(body)}, reverse=True)
    return {size: level for level, size in enumerate(sizes, start=1)}


def _render(doc: pymupdf.Document) -> TextBuilder:
    out = TextBuilder()
    blocks = _blocks(doc)
    if not blocks:
        return out
    lines = [line for block in blocks for line in block]
    body = _body_size(lines)
    levels = _levels(lines, body)
    paragraph: list[str] = []

    def flush() -> None:
        if paragraph:
            out.emit("\n".join(paragraph) + "\n\n")
            paragraph.clear()

    for block in blocks:
        for line in block:
            level = levels.get(round(line.size, 1)) if line.is_heading(body) else None
            if level is None:
                paragraph.append(line.text)
            else:
                flush()
                out.start_section(level, line.text)
                out.emit(line.text + "\n\n")
        flush()  # a block boundary is a paragraph boundary
    return out


class PdfParser:
    format_key = "pdf"
    extensions: tuple[str, ...] = (".pdf",)

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        try:
            doc = pymupdf.open(stream=data, filetype="pdf")
        except pymupdf.FileDataError as exc:
            raise PermanentIngestionError(f"{filename} is not a valid PDF file") from exc
        try:
            if doc.needs_pass:
                raise PermanentIngestionError(f"{filename} is password-protected")
            out = _render(doc)
            text, sections, code_spans = out.finish()
            if not text.strip():
                # Nothing a retry can fix: a scanned or image-only PDF.
                raise PermanentIngestionError(f"{filename} has no text")
            metadata = doc.metadata
            title = (
                text_or_none(metadata.get("title")) or out.first_h1 or title_from_filename(filename)
            )
            try:
                return ParsedDocument(
                    title=title,
                    source_format=self.format_key,
                    text=text,
                    sections=sections,
                    code_spans=code_spans,
                    metadata=DocumentMetadata(
                        tags=tags_from(metadata.get("keywords")),
                        author=text_or_none(metadata.get("author")),
                        doc_date=_doc_date(metadata.get("creationDate")),
                    ),
                )
            except ValidationError as exc:
                raise PermanentIngestionError(f"{filename} could not be parsed: {exc}") from exc
        finally:
            doc.close()
