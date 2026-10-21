"""Chunkers built on LangChain's text splitters (ADR-0013, ADR-0019).

The splitters return plain strings, but the rest of the system needs each chunk's span in
`ParsedDocument.text` (evidence coverage is measured on spans, ADR-0021). Every piece a splitter
returns is a contiguous slice of the text it was given, so `build_chunks` finds each one's
position by search. LangChain's own `add_start_index` is not used: it assumes the length
function counts characters, and it breaks when it counts tokens.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Self

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ragqa.domain.documents import Chunk, ParsedDocument
from ragqa.ingestion.chunkers.registry import TokenCounter


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FixedSizeParams(_Params):
    chunk_chars: int = Field(default=1200, gt=0)
    overlap_chars: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def _overlap_is_smaller(self) -> Self:
        if self.overlap_chars >= self.chunk_chars:
            raise ValueError("overlap_chars must be smaller than chunk_chars")
        return self


class RecursiveParams(_Params):
    chunk_tokens: int = Field(default=256, gt=0)
    overlap_tokens: int = Field(default=32, ge=0)

    @model_validator(mode="after")
    def _overlap_is_smaller(self) -> Self:
        if self.overlap_tokens >= self.chunk_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_tokens")
        return self


def _locate(text: str, piece: str, min_start: int, min_end: int) -> int:
    """Where `piece` starts in `text`.

    A piece starts after the previous piece started and ends after it ended. The second test
    skips a repeat of the same words inside the previous piece, which a plain search would
    return.
    """
    start = text.find(piece, min_start)
    while start != -1 and start + len(piece) <= min_end:
        start = text.find(piece, start + 1)
    if start == -1:
        raise ValueError("the splitter returned text that is not a piece of the document")
    return start


def build_chunks(doc: ParsedDocument, pieces: Sequence[str], counter: TokenCounter) -> list[Chunk]:
    chunks: list[Chunk] = []
    previous_start, previous_end = -1, 0
    for piece in pieces:
        if not piece.strip():
            continue
        start = _locate(doc.text, piece, previous_start + 1, previous_end)
        end = start + len(piece)
        chunks.append(
            Chunk(
                chunk_index=len(chunks),
                start=start,
                end=end,
                text=piece,
                section_path=doc.section_path_at(start),
                token_count=counter.count(piece),
                has_code=doc.overlaps_code(start, end),
            )
        )
        previous_start, previous_end = start, end
    return chunks


class FixedSizeChunker:
    """Windows of about `chunk_chars` characters that overlap by `overlap_chars`, cut at the
    nearest space. It ignores structure: the baseline of the chunking comparison (ADR-0019)."""

    strategy = "fixed-size"

    def __init__(self, params: Mapping[str, Any], counter: TokenCounter) -> None:
        p = FixedSizeParams.model_validate(params)
        self._counter = counter
        self._splitter = CharacterTextSplitter(
            separator=" ",
            keep_separator=True,  # the space stays in the text, so pieces are exact slices
            chunk_size=p.chunk_chars,
            chunk_overlap=p.overlap_chars,
            length_function=len,
        )

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        return build_chunks(doc, self._splitter.split_text(doc.text), self._counter)


class RecursiveChunker:
    """Splits on blank lines, then lines, then spaces, until each piece fits `chunk_tokens`
    tokens of the embedder's tokenizer, and packs neighbours up to that size."""

    strategy = "recursive"

    def __init__(self, params: Mapping[str, Any], counter: TokenCounter) -> None:
        p = RecursiveParams.model_validate(params)
        self._counter = counter
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=p.chunk_tokens,
            chunk_overlap=p.overlap_tokens,
            length_function=counter.count,
            keep_separator=True,
        )

    def chunk(self, doc: ParsedDocument) -> list[Chunk]:
        return build_chunks(doc, self._splitter.split_text(doc.text), self._counter)
