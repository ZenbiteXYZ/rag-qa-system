import datetime as dt
from io import BytesIO

import pytest
from docx import Document

from ragqa.domain.documents import ParsedDocument
from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.ingestion.parsers import DocxParser


def make_docx() -> bytes:
    document = Document()
    props = document.core_properties
    props.title = "Runbook"
    props.author = "Jane Doe"
    props.keywords = "ops; k8s"
    props.created = dt.datetime(2026, 3, 1, 12, 0)
    document.add_paragraph("Intro before any heading.")
    document.add_heading("Deploy", level=1)
    document.add_paragraph("Run the following command.")
    document.add_paragraph("first item", style="List Bullet")
    document.add_paragraph("second item", style="List Bullet")
    for line in ("kubectl apply -f app.yaml", "kubectl get pods"):
        document.add_paragraph().add_run(line).font.name = "Consolas"
    document.add_heading("Checks", level=2)
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Check", "Command"
    table.cell(1, 0).text, table.cell(1, 1).text = "Pods ready", "kubectl get pods"
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def doc() -> ParsedDocument:
    return DocxParser().parse(make_docx(), filename="runbook.docx")


def test_metadata_from_core_properties(doc: ParsedDocument) -> None:
    assert doc.title == "Runbook"
    assert doc.source_format == "docx"
    assert doc.metadata.author == "Jane Doe"
    assert doc.metadata.tags == ("ops", "k8s")
    assert doc.metadata.doc_date == dt.date(2026, 3, 1)


def test_sections_from_heading_styles(doc: ParsedDocument) -> None:
    assert [s.heading_path for s in doc.sections] == [(), ("Deploy",), ("Deploy", "Checks")]
    assert doc.text.startswith("Intro before any heading.")


def test_lists_code_and_tables(doc: ParsedDocument) -> None:
    assert "- first item\n- second item" in doc.text
    (code,) = doc.code_spans
    assert doc.text[code.start : code.end] == "kubectl apply -f app.yaml\nkubectl get pods"
    assert "Check | Command\nPods ready | kubectl get pods" in doc.text


def test_invalid_file_is_a_permanent_error() -> None:
    with pytest.raises(PermanentIngestionError, match="not a valid .docx"):
        DocxParser().parse(b"not a zip", filename="x.docx")


def test_empty_document_is_a_permanent_error() -> None:
    buffer = BytesIO()
    Document().save(buffer)

    with pytest.raises(PermanentIngestionError, match="no text"):
        DocxParser().parse(buffer.getvalue(), filename="x.docx")
