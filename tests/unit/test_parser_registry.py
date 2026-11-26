from pathlib import Path

import pytest

from ragqa.domain.documents import ParsedDocument
from ragqa.evaluation import text_sources
from ragqa.ingestion.errors import UnsupportedFormatError
from ragqa.ingestion.parsers import (
    DocxParser,
    MarkdownParser,
    ParserRegistry,
    PdfParser,
    default_registry,
)


class TxtParser:
    format_key = "txt"
    extensions: tuple[str, ...] = (".txt",)

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        return ParsedDocument(title=filename, source_format=self.format_key, text=data.decode())


def test_default_registry_resolves_by_extension_case_insensitively() -> None:
    registry = default_registry()

    assert isinstance(registry.for_filename("guide.MD"), MarkdownParser)
    assert isinstance(registry.for_filename("notes.docx"), DocxParser)
    assert isinstance(registry.for_filename("report.pdf"), PdfParser)
    assert registry.extensions == (".docx", ".markdown", ".md", ".pdf")


@pytest.mark.parametrize("name", ["notes.rtf", "README"])
def test_unknown_extension_is_rejected(name: str) -> None:
    with pytest.raises(UnsupportedFormatError, match="supported: .docx, .markdown, .md, .pdf"):
        default_registry().for_filename(name)


def test_a_new_format_is_one_registration() -> None:
    registry = ParserRegistry([MarkdownParser(), TxtParser()])

    doc = registry.for_filename("a.txt").parse(b"plain", filename="a.txt")

    assert doc.source_format == "txt"


def test_duplicate_extension_is_rejected() -> None:
    with pytest.raises(ValueError, match="already handled"):
        ParserRegistry([MarkdownParser(), MarkdownParser()])


def test_corpus_text_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    page = tmp_path / "release-1.33" / "concepts" / "pods.md"
    page.parent.mkdir(parents=True)
    page.write_text("# Pods\n\nA *Pod* runs containers.\n", encoding="utf-8")
    monkeypatch.setenv("RAGQA_CORPUS_DIR", str(tmp_path))

    assert (
        text_sources.corpus_text("concepts/pods.md", "release-1.33")
        == "Pods\n\nA Pod runs containers."
    )
    with pytest.raises(ValueError, match="outside"):
        text_sources.corpus_text("../../etc/passwd", "release-1.33")
    with pytest.raises(ValueError, match="no parser"):
        (tmp_path / "release-1.33" / "x.txt").write_bytes(b"plain")
        text_sources.corpus_text("x.txt", "release-1.33")
