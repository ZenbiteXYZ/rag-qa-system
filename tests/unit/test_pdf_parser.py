import datetime as dt

import pymupdf
import pytest

from ragqa.domain.documents import Chunk, ParsedDocument, check_chunks, normalize_text
from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.ingestion.parsers import PdfParser


def make_pdf(lines: list[tuple[str, float, str]], metadata: dict[str, str] | None = None) -> bytes:
    """A one-page PDF with one line of text per (text, fontsize, fontname) tuple."""
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72.0
    for text, size, fontname in lines:
        page.insert_text((50, y), text, fontsize=size, fontname=fontname)
        y += size + 12
    if metadata:
        doc.set_metadata(metadata)
    return doc.tobytes()


def parse(lines: list[tuple[str, float, str]], filename: str = "guide.pdf") -> ParsedDocument:
    return PdfParser().parse(make_pdf(lines), filename=filename)


def test_text_is_extracted_and_normalized() -> None:
    # One insert_text call: both lines land in one text block, so they are one paragraph.
    doc = parse([("Run the command.\nVerify the pods.", 10, "helv")])

    assert doc.source_format == "pdf"
    assert "Run the command.\nVerify the pods." in doc.text
    assert doc.text == normalize_text(doc.text)
    assert doc.code_spans == ()


def test_sections_come_from_font_sizes() -> None:
    doc = parse(
        [
            ("Deploy", 20, "hebo"),
            ("Run the following command.", 10, "helv"),
            ("Check the pods.", 10, "helv"),
            ("Checks", 14, "hebo"),
            ("Pods are ready.", 10, "helv"),
        ],
        filename="runbook.pdf",
    )

    assert [s.heading_path for s in doc.sections] == [("Deploy",), ("Deploy", "Checks")]
    assert [s.level for s in doc.sections] == [1, 2]
    for section in doc.sections:
        assert doc.text[section.start : section.end].strip()
    assert doc.section_path_at(doc.sections[0].start + 10) == ("Deploy",)
    assert doc.section_path_at(doc.sections[1].start + 2) == ("Deploy", "Checks")


def test_a_document_without_headings_is_one_flat_section() -> None:
    doc = parse([("Only body text.", 10, "helv"), ("More text.", 10, "helv")])

    (section,) = doc.sections
    assert section.heading_path == ()
    assert section.level == 0
    assert section.start == 0
    assert doc.text[section.start : section.end].startswith("Only body text.")


def test_a_pdf_without_text_is_a_permanent_error() -> None:
    doc = pymupdf.open()  # a page with no text: a scanned or image-only PDF
    doc.new_page()
    data = doc.tobytes()

    with pytest.raises(PermanentIngestionError, match="no text"):
        PdfParser().parse(data, filename="scan.pdf")


def test_a_password_protected_pdf_is_a_permanent_error() -> None:
    doc = pymupdf.open()
    doc.new_page()
    data = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret")

    with pytest.raises(PermanentIngestionError, match="password"):
        PdfParser().parse(data, filename="locked.pdf")


def test_invalid_bytes_are_a_permanent_error() -> None:
    with pytest.raises(PermanentIngestionError, match="not a valid PDF"):
        PdfParser().parse(b"clearly not a pdf", filename="broken.pdf")


def test_metadata_comes_from_the_info_dictionary() -> None:
    data = make_pdf(
        [("Body.", 10, "helv")],
        metadata={
            "title": "Runbook",
            "author": "Jane Doe",
            "keywords": "ops, k8s",
            "creationDate": "D:20260301120000Z",
        },
    )
    doc = PdfParser().parse(data, filename="runbook.pdf")

    assert doc.title == "Runbook"
    assert doc.metadata.author == "Jane Doe"
    assert doc.metadata.tags == ("ops", "k8s")
    assert doc.metadata.doc_date == dt.date(2026, 3, 1)
    assert doc.metadata.language is None


def test_title_falls_back_to_the_first_heading_then_the_filename() -> None:
    with_heading = make_pdf(
        [("Deploy", 20, "hebo"), ("Run the following command.\nCheck the pods.", 10, "helv")]
    )
    assert PdfParser().parse(with_heading, filename="runbook.pdf").title == "Deploy"

    without_headings = make_pdf([("Body text.", 10, "helv")])
    assert PdfParser().parse(without_headings, filename="my-report.pdf").title == "my report"


def test_an_unparseable_creation_date_is_ignored() -> None:
    data = make_pdf([("Body.", 10, "helv")], metadata={"creationDate": "D:nonsense"})

    assert PdfParser().parse(data, filename="x.pdf").metadata.doc_date is None


def test_windows_over_the_text_satisfy_the_chunk_invariants() -> None:
    doc = parse(
        [
            ("Deploy", 20, "hebo"),
            ("Run the following command.", 10, "helv"),
            ("Check the pods.", 10, "helv"),
        ],
        filename="runbook.pdf",
    )

    chunks = [
        Chunk(
            start=start,
            end=end,
            chunk_index=index,
            text=doc.text[start:end],
            section_path=doc.section_path_at(start),
            token_count=end - start,
            has_code=doc.overlaps_code(start, end),
        )
        for index, start in enumerate(range(0, len(doc.text), 32))
        for end in (min(start + 32, len(doc.text)),)
    ]

    check_chunks(doc, chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.section_path == ("Deploy",) for c in chunks)
    assert not any(c.has_code for c in chunks)
