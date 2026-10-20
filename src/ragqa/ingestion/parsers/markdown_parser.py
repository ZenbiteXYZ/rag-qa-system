"""Markdown parser (ADR-0021).

The parsed text is close to what a reader sees on the rendered page:
- front matter is removed and mapped to metadata,
- headings stay as plain lines and define the sections,
- inline markup is stripped (emphasis, links keep their text, images keep their alt text),
- Hugo shortcodes are removed (`glossary_tooltip` keeps its `text`), except inside code,
- code blocks and tables keep their content, and code blocks are recorded as code spans.
"""

import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

import yaml  # type: ignore[import-untyped]
from markdown_it import MarkdownIt
from markdown_it.token import Token
from pydantic import ValidationError

from ragqa.domain.documents import DocumentMetadata, ParsedDocument, normalize_text
from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.ingestion.parsers.builder import TextBuilder
from ragqa.ingestion.parsers.common import tags_from, text_or_none, title_from_filename
from ragqa.observability.logging import get_logger

log = get_logger(__name__)

_FRONT_MATTER = re.compile(r"\A---[ \t]*\n(.*?\n)?(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.DOTALL)
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_SHORTCODE = re.compile(r"\{\{[<%]\s*(/?)\s*([\w.-]+)(.*?)\s*[>%]\}\}", re.DOTALL)
_PARAM = r"""\b{}\s*=\s*(?:"([^"]*)"|'([^']*)')"""
_TEXT_PARAM = re.compile(_PARAM.format("text"))
# The site shows the glossary entry's title; without the glossary, the term id is the
# closest stand-in (for example "deployment" for "Deployment").
_TERM_PARAM = re.compile(_PARAM.format("term_id"))
_TEXT_SHORTCODES = {"glossary_tooltip", "tooltip"}  # rendered as their `text` parameter
_HEADING_ID = re.compile(r"\s*\{#[\w.:-]+\}\s*$")  # Hugo heading anchors: "Title {#id}"
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_TAG = re.compile(r"</?[A-Za-z][^>]*>")

_md = MarkdownIt("commonmark").enable(["table", "strikethrough"])


def split_front_matter(source: str) -> tuple[dict[str, Any], str]:
    """Return the front matter mapping and the rest. A block that is not valid YAML or not a
    mapping (for example a horizontal rule) is left in the text."""
    match = _FRONT_MATTER.match(source)
    if match is None:
        return {}, source
    try:
        data = yaml.safe_load(match.group(1) or "")
    except yaml.YAMLError:
        log.warning("front_matter_invalid_yaml")
        return {}, source
    if data is None:
        data = {}
    if not isinstance(data, dict):
        return {}, source
    return {str(k): v for k, v in data.items()}, source[match.end() :]


def _replace_shortcode(match: re.Match[str]) -> str:
    closing, name, params = match.groups()
    if closing or name not in _TEXT_SHORTCODES:
        return ""
    param = _TEXT_PARAM.search(params) or _TERM_PARAM.search(params)
    return (param.group(1) or param.group(2) or "") if param else ""


def map_prose(source: str, transform: Callable[[str], str]) -> str:
    """Apply `transform` to the text outside fenced code blocks. Code is left untouched."""
    segments: list[str] = []
    prose: list[str] = []
    fence: str | None = None

    def flush() -> None:
        if prose:
            segments.append(transform("\n".join(prose)))
            prose.clear()

    for line in source.split("\n"):
        marker = _FENCE.match(line)
        if fence is None:
            if marker:
                flush()
                segments.append(line)
                fence = marker.group(1)
            else:
                prose.append(line)
        else:
            segments.append(line)
            if marker and line.strip() == marker.group(1):
                same_kind = marker.group(1)[0] == fence[0]
                if same_kind and len(marker.group(1)) >= len(fence):
                    fence = None
    flush()
    return "\n".join(segments)


def strip_shortcodes(source: str) -> str:
    """Remove Hugo shortcodes from prose. Fenced code blocks are left untouched."""
    return map_prose(source, lambda prose: _SHORTCODE.sub(_replace_shortcode, prose))


def _inline_piece(token: Token) -> str:
    if token.type in {"text", "code_inline", "text_special"}:
        return token.content
    if token.type == "softbreak":
        return " "
    if token.type == "hardbreak":
        return "\n"
    if token.type == "image":
        return token.content  # alt text
    return ""  # emphasis and link tags, inline HTML


def _inline_text(token: Token) -> str:
    return "".join(_inline_piece(child) for child in token.children or [])


def _html_text(content: str) -> str:
    return _HTML_TAG.sub("", _HTML_COMMENT.sub("", content)).strip()


def _fence_language(info: str) -> str | None:
    words = info.strip().split()
    return words[0].strip("{}") or None if words else None


def _render(tokens: Sequence[Token]) -> TextBuilder:
    out = TextBuilder()
    lists: list[list[int | bool]] = []  # [ordered, next number] per open list
    pending_marker: str | None = None
    row: list[str] | None = None

    def sep() -> str:
        return "\n" if lists else "\n\n"

    i = 0
    while i < len(tokens):
        token = tokens[i]
        kind = token.type
        if kind == "heading_open":
            title = _HEADING_ID.sub("", _inline_text(tokens[i + 1])).strip()
            if title:
                out.start_section(int(token.tag[1]), title)
                out.emit(title + "\n\n")
            i += 3  # heading_open, inline, heading_close
            continue
        if kind in {"bullet_list_open", "ordered_list_open"}:
            lists.append([kind == "ordered_list_open", int(token.attrGet("start") or 1)])
        elif kind in {"bullet_list_close", "ordered_list_close"}:
            lists.pop()
            if not lists:
                out.emit("\n")
        elif kind == "list_item_open":
            ordered, number = lists[-1]
            marker = f"{number}. " if ordered else "- "
            lists[-1][1] = int(number) + 1
            pending_marker = "  " * (len(lists) - 1) + marker
        elif kind == "list_item_close":
            pending_marker = None
        elif kind == "paragraph_open":
            text = _inline_text(tokens[i + 1])
            if text.strip():
                prefix = pending_marker if pending_marker is not None else "  " * len(lists)
                out.emit(prefix + text + sep())
            pending_marker = None
            i += 3  # paragraph_open, inline, paragraph_close
            continue
        elif kind in {"fence", "code_block"}:
            content = token.content.rstrip("\n")
            if content.strip():
                language = _fence_language(token.info) if kind == "fence" else None
                out.add_code(content, language)
                out.emit(sep())
        elif kind == "html_block":
            text = _html_text(token.content)
            if text:
                out.emit(text + sep())
        elif kind == "tr_open":
            row = []
        elif kind == "inline" and row is not None:
            row.append(_inline_text(token).strip())
        elif kind == "tr_close" and row is not None:
            out.emit(" | ".join(row) + "\n")
            row = None
        elif kind == "table_close":
            out.emit("\n")
        i += 1
    return out


def _json_safe(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(json.dumps(value, default=str))
    return result


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10])
        except ValueError:
            return None
    return None


def metadata_from_front_matter(front: Mapping[str, Any]) -> DocumentMetadata:
    return DocumentMetadata(
        doc_type=text_or_none(front.get("content_type") or front.get("doc_type")),
        tags=tags_from(front.get("tags")),
        author=text_or_none(front.get("author")),
        doc_date=_date(front.get("date") or front.get("doc_date")),
        language=text_or_none(front.get("language") or front.get("lang")),
    )


class MarkdownParser:
    format_key = "md"
    extensions: tuple[str, ...] = (".md", ".markdown")

    def parse(self, data: bytes, *, filename: str) -> ParsedDocument:
        try:
            source = data.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise PermanentIngestionError(f"{filename} is not valid UTF-8") from exc

        front, body = split_front_matter(normalize_text(source))
        out = _render(_md.parse(strip_shortcodes(body)))
        text, sections, code_spans = out.finish()
        if not text.strip():
            raise PermanentIngestionError(f"{filename} has no text")

        title = text_or_none(front.get("title")) or out.first_h1 or title_from_filename(filename)
        try:
            return ParsedDocument(
                title=title,
                source_format=self.format_key,
                text=text,
                sections=sections,
                code_spans=code_spans,
                metadata=metadata_from_front_matter(front),
                front_matter=_json_safe(front),
            )
        except ValidationError as exc:
            raise PermanentIngestionError(f"{filename} could not be parsed: {exc}") from exc
