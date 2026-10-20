"""Assembles `ParsedDocument.text` and its spans while a parser walks a document.

Parsers emit text in reading order. The builder tracks offsets, so sections and code
blocks are recorded as spans into the very string it returns (ADR-0021).
"""

from ragqa.domain.documents import CodeSpan, Section


class TextBuilder:
    def __init__(self) -> None:
        self._parts: list[str] = []
        self.pos = 0
        self._content_end = 0  # end of the last non-newline character
        self._stack: list[tuple[int, str]] = []  # (level, title) of the open headings
        self._open: tuple[int, tuple[str, ...], int] | None = None  # start, path, level
        self._sections: list[Section] = []
        self._code: list[CodeSpan] = []
        self.first_h1: str | None = None

    def emit(self, text: str) -> None:
        stripped = text.rstrip("\n")
        if stripped:
            self._content_end = self.pos + len(stripped)
        self._parts.append(text)
        self.pos += len(text)

    def start_section(self, level: int, title: str) -> None:
        """Call just before emitting the heading's text."""
        if self._open is None and not self._sections and self._content_end > 0:
            # Text before the first heading becomes its own section.
            self._sections.append(Section(start=0, end=self._content_end))
        self._close_section()
        while self._stack and self._stack[-1][0] >= level:
            self._stack.pop()
        self._stack.append((level, title))
        self._open = (self.pos, tuple(t for _, t in self._stack), level)
        if level == 1 and self.first_h1 is None:
            self.first_h1 = title

    def add_code(self, content: str, language: str | None) -> None:
        """Emit a code block's text and record its span. No separator is added."""
        start = self.pos
        self.emit(content)
        self._code.append(CodeSpan(start=start, end=start + len(content), language=language))

    def _close_section(self) -> None:
        if self._open is None:
            return
        start, path, level = self._open
        if self._content_end > start:
            self._sections.append(
                Section(start=start, end=self._content_end, heading_path=path, level=level)
            )
        self._open = None

    def finish(self) -> tuple[str, tuple[Section, ...], tuple[CodeSpan, ...]]:
        if self._open is None and not self._sections and self._content_end > 0:
            self._sections.append(Section(start=0, end=self._content_end))  # no headings at all
        self._close_section()
        text = "".join(self._parts).rstrip("\n")
        return text, tuple(self._sections), tuple(self._code)
