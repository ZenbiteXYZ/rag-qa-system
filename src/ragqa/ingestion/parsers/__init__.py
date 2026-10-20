"""Document parsers and their registry."""

from ragqa.ingestion.parsers.docx_parser import DocxParser
from ragqa.ingestion.parsers.markdown_parser import MarkdownParser
from ragqa.ingestion.parsers.registry import Parser, ParserRegistry


def default_registry() -> ParserRegistry:
    """The formats supported in v1 (ADR-0001): Markdown and Word."""
    return ParserRegistry([MarkdownParser(), DocxParser()])


__all__ = ["DocxParser", "MarkdownParser", "Parser", "ParserRegistry", "default_registry"]
