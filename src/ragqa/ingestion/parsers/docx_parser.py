"""Word (.docx) parser (ADR-0021).

Headings come from the paragraph styles `Heading 1` to `Heading 9`. List paragraphs become
`- ` lines, tables become one `a | b` line per row, and consecutive code paragraphs (a
code style or a monospace font) are merged into one code span. Images, text boxes,
headers, footers and footnotes are ignored.
"""

import re
from collections.abc import Iterator
from io import BytesIO
from zipfile import BadZipFile

from docx import Document
from docx.document import Document as DocumentType
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from pydantic import ValidationError

from ragqa.domain.documents import DocumentMetadata, ParsedDocument, normalize_text
from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.ingestion.parsers.builder import TextBuilder
from ragqa.ingestion.parsers.common import tags_from, text_or_none, title_from_filename

_HEADING = re.compile(r"^Heading (\d)$")
_CODE_STYLE = re.compile(r"code|preformatted|monospace", re.IGNORECASE)
_MONOSPACE = {
    "courier new",
    "courier",
    "consolas",
    "lucida console",
    "menlo",
    "monaco",
    "source code pro",
}


def _blocks(document: DocumentType) -> Iterator[Paragraph | Table]:
    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield Table(child, document)


def _heading_level(paragraph: Paragraph) -> int | None:
    match = _HEADING.match(paragraph.style.name if paragraph.style is not None else "")
    return int(match.group(1)) if match else None


def _is_code(paragraph: Paragraph) -> bool:
    if paragraph.style is not None and _CODE_STYLE.search(paragraph.style.name or ""):
        return True
    runs = [r for r in paragraph.runs if r.text.strip()]
    fonts = [(r.font.name or "").lower() for r in runs]
    return bool(runs) and all(font in _MONOSPACE for font in fonts)


def _list_level(paragraph: Paragraph) -> int | None:
    """Nesting level of a list paragraph, or None if it is not in a list."""
    if paragraph._p.xpath("./w:pPr/w:numPr"):
        levels = paragraph._p.xpath("./w:pPr/w:numPr/w:ilvl/@w:val")
        return int(levels[0]) if levels else 0
    if paragraph.style is not None and (paragraph.style.name or "").startswith("List"):
        return 0
    return None


def _table_lines(table: Table) -> list[str]:
    lines = []
    for row in table.rows:
        cells = []
        previous = None
        for cell in row.cells:
            if cell._tc is previous:  # a merged cell appears once per column it spans
                continue
            previous = cell._tc
            cells.append(" ".join(normalize_text(cell.text).split()))
        if any(cells):
            lines.append(" | ".join(cells))
    return lines


def _render(document: DocumentType) -> TextBuilder:
    out = TextBuilder()
    code_lines: list[str] = []
    previous_list = False

    def flush_code() -> None:
        if code_lines:
            out.add_code("\n".join(code_lines), None)
            out.emit("\n\n")
            code_lines.clear()

    for block in _blocks(document):
        if isinstance(block, Table):
            flush_code()
            lines = _table_lines(block)
            if lines:
                out.emit("\n".join(lines) + "\n\n")
            previous_list = False
            continue

        text = normalize_text(block.text).strip("\n")
        if not text.strip():
            continue
        if _is_code(block):
            code_lines.append(text.rstrip())
            continue
        flush_code()

        level = _heading_level(block)
        list_level = _list_level(block)
        if level is not None:
            out.start_section(level, text.strip())
            out.emit(text.strip() + "\n\n")
        elif list_level is not None:
            out.emit("  " * list_level + "- " + text.strip() + "\n")
        else:
            if previous_list:
                out.emit("\n")
            out.emit(text.strip() + "\n\n")
        previous_list = list_level is not None and level is None
    flush_code()
    return out


def _metadata(document: DocumentType) -> DocumentMetadata:
    props = document.core_properties
    created = props.created
    return DocumentMetadata(
        tags=tags_from((props.keywords or "").replace(";", ",")),
        author=text_or_none(props.author),
        doc_date=created.date() if created is not None else None,
        language=text_or_none(props.language),
    )


class DocxParser:
    format_key = "docx"
    extensions: tuple[str, ...] = (".docx",)

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        try:
            document = Document(BytesIO(data))
        except (PackageNotFoundError, BadZipFile, KeyError, ValueError) as exc:
            raise PermanentIngestionError(f"{filename} is not a valid .docx file") from exc

        out = _render(document)
        text, sections, code_spans = out.finish()
        if not text.strip():
            raise PermanentIngestionError(f"{filename} has no text")

        title = (
            text_or_none(document.core_properties.title)
            or out.first_h1
            or title_from_filename(filename)
        )
        try:
            return ParsedDocument(
                title=title,
                source_format=self.format_key,
                text=text,
                sections=sections,
                code_spans=code_spans,
                metadata=_metadata(document),
            )
        except ValidationError as exc:
            raise PermanentIngestionError(f"{filename} could not be parsed: {exc}") from exc
