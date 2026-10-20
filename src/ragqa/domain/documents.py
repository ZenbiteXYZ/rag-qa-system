"""Document model shared by parsers, chunkers and storage (ADR-0021).

A parser turns one document version into a `ParsedDocument`: one normalized string plus
spans that point into it. Sections, code blocks, chunks and the golden-set evidence all
use character offsets into that same string, so no stage re-joins text its own way.
"""

import unicodedata
from collections.abc import Sequence
from datetime import date
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


def normalize_text(text: str) -> str:
    """NFC form and `\\n` line endings. Every parser must return text in this form."""
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _Spanned(_Model):
    """A character range `[start, end)`."""

    start: int = Field(ge=0)
    end: int

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        if self.end <= self.start:
            raise ValueError(f"empty or reversed span [{self.start}, {self.end})")
        return self


class DocumentMetadata(_Model):
    doc_type: str | None = None
    tags: tuple[str, ...] = ()
    author: str | None = None
    doc_date: date | None = None
    language: str | None = None


class Section(_Spanned):
    """A heading and the text under it, up to the next heading."""

    heading_path: tuple[str, ...] = ()  # empty for text before the first heading
    level: int = Field(default=0, ge=0)  # heading depth, 0 for that preamble


class CodeSpan(_Spanned):
    """A code block (for example a fenced block in markdown)."""

    language: str | None = None


def _check_spans(spans: Sequence[_Spanned], limit: int, what: str) -> None:
    """Spans must lie inside the text, in order, without overlapping."""
    previous_end = 0
    for span in spans:
        if span.end > limit:
            raise ValueError(f"{what} [{span.start}, {span.end}) extends past the text ({limit})")
        if span.start < previous_end:
            raise ValueError(f"{what}s must be in order and not overlap, at {span.start}")
        previous_end = span.end


class ParsedDocument(_Model):
    title: str
    # Key of the parser in the parser registry, for example "md". Not a closed set here, so
    # adding a format never changes this model.
    source_format: str = Field(pattern=r"^[a-z0-9]+$")
    text: str = Field(min_length=1)
    sections: tuple[Section, ...] = ()
    code_spans: tuple[CodeSpan, ...] = ()
    metadata: DocumentMetadata = DocumentMetadata()
    # Raw front matter, kept for debugging. Nothing downstream reads it.
    front_matter: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_text_and_spans(self) -> Self:
        if self.text != normalize_text(self.text):
            raise ValueError("text must be NFC with \\n line endings (use normalize_text)")
        _check_spans(self.sections, len(self.text), "section")
        _check_spans(self.code_spans, len(self.text), "code span")
        return self

    def section_path_at(self, offset: int) -> tuple[str, ...]:
        """Heading path of the section that contains `offset`, empty before the first one."""
        path: tuple[str, ...] = ()
        for section in self.sections:
            if section.start > offset:
                break
            path = section.heading_path
        return path

    def overlaps_code(self, start: int, end: int) -> bool:
        """True if any code block shares a character with `[start, end)`."""
        return any(c.start < end and start < c.end for c in self.code_spans)


class Chunk(_Spanned):
    """A piece of a document. `text` is what gets embedded and indexed for BM25."""

    chunk_index: int = Field(ge=0)
    # May differ from `doc.text[start:end]`, for example with a heading breadcrumb in front.
    # Evidence matching always uses the span.
    text: str = Field(min_length=1)
    section_path: tuple[str, ...] = ()  # of the section where the chunk starts
    token_count: int = Field(gt=0)  # counted with the embedder's tokenizer
    has_code: bool = False


class EmbeddedChunk(_Model):
    """A chunk with its dense vector. The sparse BM25 vector is built by Qdrant from the text."""

    chunk: Chunk
    dense: tuple[float, ...] = Field(min_length=1)

    @classmethod
    def of(cls, chunk: Chunk, dense: Sequence[float], *, size: int) -> Self:
        """Build with a length check against the index's `dense_vector_size`."""
        if len(dense) != size:
            raise ValueError(f"dense vector has {len(dense)} values, the index expects {size}")
        return cls(chunk=chunk, dense=tuple(dense))


def check_chunks(doc: ParsedDocument, chunks: Sequence[Chunk]) -> None:
    """Invariants over a chunker's whole output for one document."""
    if not chunks:
        raise ValueError("a chunker returned no chunks")
    for expected, chunk in enumerate(chunks):
        if chunk.chunk_index != expected:
            raise ValueError(
                f"chunk_index must run 0, 1, 2, ...; got {chunk.chunk_index} at {expected}"
            )
        if chunk.end > len(doc.text):
            raise ValueError(f"chunk {expected} extends past the document text ({len(doc.text)})")
