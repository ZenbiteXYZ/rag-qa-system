import unicodedata

import pytest
from pydantic import ValidationError

from ragqa.domain.documents import (
    Chunk,
    CodeSpan,
    DocumentMetadata,
    EmbeddedChunk,
    ParsedDocument,
    Section,
    check_chunks,
    normalize_text,
)

TEXT = "Pods\nA Pod runs containers.\nUsing Pods\nRun kubectl get pods.\nkubectl get pods\n"


def make_doc(**overrides: object) -> ParsedDocument:
    fields: dict[str, object] = {
        "title": "Pods",
        "source_format": "md",
        "text": TEXT,
        "sections": [
            Section(start=0, end=28, heading_path=("Pods",), level=1),
            Section(start=28, end=len(TEXT), heading_path=("Pods", "Using Pods"), level=2),
        ],
        "code_spans": [CodeSpan(start=len(TEXT) - 17, end=len(TEXT) - 1, language="sh")],
        "metadata": DocumentMetadata(doc_type="concept", tags=("pods",)),
    }
    fields.update(overrides)
    return ParsedDocument(**fields)  # type: ignore[arg-type]


def make_chunk(index: int = 0, start: int = 0, end: int = 10, **overrides: object) -> Chunk:
    fields: dict[str, object] = {
        "chunk_index": index,
        "start": start,
        "end": end,
        "text": "chunk text",
        "token_count": 3,
    }
    fields.update(overrides)
    return Chunk(**fields)  # type: ignore[arg-type]


def test_valid_document_keeps_everything_as_tuples() -> None:
    doc = make_doc()

    assert isinstance(doc.sections, tuple)
    assert doc.metadata.tags == ("pods",)
    assert doc.source_format == "md"


def test_normalize_text_uses_nfc_and_unix_newlines() -> None:
    decomposed = unicodedata.normalize("NFD", "café")

    assert normalize_text("a\r\nb\rc") == "a\nb\nc"
    assert normalize_text(decomposed) == "café"


@pytest.mark.parametrize("bad", ["a\r\nb", unicodedata.normalize("NFD", "café")])
def test_document_text_must_be_normalized(bad: str) -> None:
    with pytest.raises(ValidationError, match="NFC"):
        make_doc(text=bad, sections=[], code_spans=[])


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValidationError):
        make_doc(text="", sections=[], code_spans=[])


@pytest.mark.parametrize("fmt", ["PDF", "my format", ""])
def test_source_format_is_a_registry_key(fmt: str) -> None:
    with pytest.raises(ValidationError):
        make_doc(source_format=fmt)


def test_new_formats_need_no_model_change() -> None:
    assert make_doc(source_format="pptx").source_format == "pptx"


def test_sections_must_be_ordered_and_not_overlap() -> None:
    overlapping = [Section(start=0, end=20), Section(start=10, end=30)]

    with pytest.raises(ValidationError, match="not overlap"):
        make_doc(sections=overlapping, code_spans=[])


def test_spans_must_lie_inside_the_text() -> None:
    with pytest.raises(ValidationError, match="past the text"):
        make_doc(sections=[Section(start=0, end=len(TEXT) + 1)], code_spans=[])


@pytest.mark.parametrize("start,end", [(5, 5), (7, 3), (-1, 4)])
def test_span_must_be_non_empty_and_forward(start: int, end: int) -> None:
    with pytest.raises(ValidationError):
        Section(start=start, end=end)


def test_models_are_frozen_and_reject_unknown_fields() -> None:
    doc = make_doc()

    with pytest.raises(ValidationError):
        doc.title = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Section(start=0, end=1, surprise=True)  # type: ignore[call-arg]


def test_section_path_at() -> None:
    doc = make_doc()

    assert doc.section_path_at(0) == ("Pods",)
    assert doc.section_path_at(27) == ("Pods",)
    assert doc.section_path_at(28) == ("Pods", "Using Pods")
    assert make_doc(sections=[], code_spans=[]).section_path_at(5) == ()


def test_overlaps_code() -> None:
    doc = make_doc()
    code_start = len(TEXT) - 17

    assert doc.overlaps_code(code_start - 5, code_start + 1)
    assert not doc.overlaps_code(0, code_start)
    assert not doc.overlaps_code(len(TEXT) - 1, len(TEXT))  # touches the end only


def test_chunk_rejects_empty_text_and_zero_tokens() -> None:
    with pytest.raises(ValidationError):
        make_chunk(text="")
    with pytest.raises(ValidationError):
        make_chunk(token_count=0)


def test_embedded_chunk_checks_vector_length() -> None:
    chunk = make_chunk()

    ok = EmbeddedChunk.of(chunk, [0.1, 0.2, 0.3], size=3)

    assert ok.dense == (0.1, 0.2, 0.3)
    with pytest.raises(ValueError, match="expects 4"):
        EmbeddedChunk.of(chunk, [0.1, 0.2, 0.3], size=4)
    with pytest.raises(ValidationError):
        EmbeddedChunk(chunk=chunk, dense=())


def test_check_chunks_accepts_overlapping_chunks_in_order() -> None:
    doc = make_doc()
    chunks = [make_chunk(0, 0, 30), make_chunk(1, 20, 60)]

    check_chunks(doc, chunks)


def test_check_chunks_rejects_gaps_and_out_of_range() -> None:
    doc = make_doc()

    with pytest.raises(ValueError, match="no chunks"):
        check_chunks(doc, [])
    with pytest.raises(ValueError, match="0, 1, 2"):
        check_chunks(doc, [make_chunk(0), make_chunk(2)])
    with pytest.raises(ValueError, match="past the document"):
        check_chunks(doc, [make_chunk(0, 0, len(TEXT) + 1)])


def test_json_round_trip() -> None:
    doc = make_doc(front_matter={"title": "Pods", "weight": 10})

    restored = ParsedDocument.model_validate_json(doc.model_dump_json())

    assert restored == doc
