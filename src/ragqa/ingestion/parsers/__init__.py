"""Document parsers and their registry."""

from ragqa.ingestion.parsers.docx_parser import DocxParser
from ragqa.ingestion.parsers.markdown_parser import MarkdownParser
from ragqa.ingestion.parsers.pdf_parser import PdfParser
from ragqa.ingestion.parsers.registry import Parser, ParserRegistry


def default_registry() -> ParserRegistry:
    """The supported formats: Markdown and Word (ADR-0001), PDF (ADR-0024)."""
    return ParserRegistry([MarkdownParser(), DocxParser(), PdfParser()])


__all__ = [
    "DocxParser",
    "MarkdownParser",
    "Parser",
    "ParserRegistry",
    "PdfParser",
    "default_registry",
]
