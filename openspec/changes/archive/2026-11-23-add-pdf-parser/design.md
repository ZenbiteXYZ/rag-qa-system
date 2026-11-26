# Design

## Context

See proposal.md - Why. The parser contract is fixed by ADR-0021: a `Parser` exposes `format_key`, `extensions` and `parse(data: bytes, *, filename: str) -> ParsedDocument`, and `default_registry()` in `src/ragqa/ingestion/parsers/__init__.py` registers parsers by extension. `DocxParser` is the closest template: it walks blocks, renders through `TextBuilder` (which tracks offsets and produces sections and code spans), maps core properties to `DocumentMetadata`, and raises `PermanentIngestionError` for unparseable or empty input. The owner chose PyMuPDF.

## Goals / Non-Goals

**Goals:**
- Accept text-based `.pdf` uploads and produce a conforming `ParsedDocument` (NFC, `\n`, valid spans).
- Best-effort sections from font cues; metadata from the PDF Info dictionary.
- Fail permanently and clearly on scanned, empty or encrypted PDFs.

**Non-Goals:**
- OCR for scanned PDFs (a later, separate decision).
- Code-span detection for PDF (no reliable semantic marker).
- Any change to the frozen evaluation protocol, the golden set, the domain model, the API, chunkers or storage.

## Decisions

### Library: PyMuPDF
Use `pymupdf` (import `pymupdf`, not the legacy `fitz` alias). Open bytes with `pymupdf.open(stream=data, filetype="pdf")`. It exposes per-span font size and flags via `page.get_text("dict")`, the Info dictionary via `doc.metadata`, and can build PDFs in memory for tests. License is AGPL-3.0, accepted by the owner. Alternatives considered: `pdfplumber` (MIT, strong tables but slower and the same custom heading logic) and `pypdf` (BSD, weak layout). `pymupdf4llm` (PDF to structured Markdown) was noted as a future option that could reuse the markdown parser, but rejected for now to keep one dependency and explicit control over spans.

### Heading inference (best-effort)
PDF has no heading semantics. From `page.get_text("dict")` blocks/lines/spans, collect each span's `size` and `flags` (bit 4 bold, bit 3 monospaced). Determine the body size as the most common size weighted by text length. Treat a line as a heading when its dominant span size is clearly larger than body (or bold and larger), is short, and is not monospaced body code. Rank the distinct heading sizes descending into levels 1..n and feed `TextBuilder.start_section(level, title)` before emitting the heading text. When no line qualifies, emit plain paragraphs and let `TextBuilder.finish()` produce one flat section. This is deliberately heuristic; false positives are acceptable and documented. Alternative considered: PDF structure tags (`H1`/`H2`) when present, used as a hint but not required since most PDFs are untagged.

### Text assembly and normalization
Walk pages in order, emit heading and paragraph text through `TextBuilder`, separating blocks with blank lines. Run all extracted text through `normalize_text` (NFC, `\n`). No code spans. The title comes from `doc.metadata["title"]`, else the first detected level-1 heading (`TextBuilder.first_h1`), else `title_from_filename`.

### Metadata
Map `doc.metadata` to `DocumentMetadata`: `author` from `author`, `doc_date` parsed from `creationDate` (PDF date string `D:YYYYMMDD...`), `tags` from `keywords` via `tags_from`, `language` unset (PDF Info has no reliable language). Use `text_or_none` for strings. `doc_date` parsing failures yield None, not an error.

### Failure handling
Raise `PermanentIngestionError` when: the bytes are not a valid PDF (open raises), `doc.needs_pass` is true (encrypted), or the assembled text is empty/whitespace (scanned or image-only). These match the docx parser's contract so the worker fails the job at once (ADR-0011).

### Registration
Add `PdfParser()` to `default_registry()`, export it in `__all__`, and update the module docstring (it currently says "formats supported in v1 (ADR-0001): Markdown and Word"). The API picks up `.pdf` automatically through the registry; no API change.

### ADR-0024 (written during apply)
Append to `docs/ADR.md`, dated 2026-11-23, format Status/Context/Decision/Consequences:
- Status: Accepted. Amends the scope of ADR-0001.
- Context: ADR-0001 excluded PDF for v1 because layout extraction is unreliable; ADR-0021's registry makes adding a format a parser-only change. Users need PDF ingestion.
- Decision: Support text-based `.pdf` via a `PdfParser` on PyMuPDF (AGPL-3.0). Extract text into the normalized span contract; infer sections best-effort from font size/bold; map the Info dictionary to metadata. Reject scanned/image-only and password-protected PDFs as permanent errors. No OCR, no code spans. TXT and HTML remain out of scope.
- Consequences: Section quality is best-effort and varies by PDF; the AGPL dependency is accepted; the evaluation protocol and corpus are unchanged. OCR and structure-tag fidelity are future work.

## Risks / Trade-offs

- [AGPL-3.0 dependency] -> Owner accepted; note it in ADR-0024 and the proposal. Revisit if distribution terms change.
- [Unreliable heading detection] -> Best-effort heuristic with a flat-section fallback; documented as such, never blocks ingestion.
- [Scanned PDFs rejected] -> Clear permanent error; OCR deferred to a separate decision.
- [Layout quirks (multi-column, headers/footers)] -> Reading-order extraction via `get_text("dict")`; accept imperfection for v1, consistent with ADR-0001's original concern.

## Migration Plan

1. Write ADR-0024 in `docs/ADR.md`.
2. Add `pymupdf` to `pyproject.toml`; `uv sync --extra dev`.
3. Implement `pdf_parser.py`, register and export it.
4. Add unit tests (build PDFs in memory with `pymupdf.open()` / `insert_text` / `set_metadata` / `tobytes()`).
5. `make check` clean; archive the change to sync the `document-model` spec.
Rollback: remove `PdfParser` from `default_registry()` and the dependency; `.pdf` uploads reject again. No data migration.

## Open Questions

- Whether to render detected tables in v1 or defer. Resolved during apply (2026-11-23): tables are not rendered in v1. `find_tables` adds runtime on untagged PDFs and risks noise in the reading order, and no spec scenario or task depends on table rendering. The spec delta is unchanged (it says nothing about tables). Revisit as a separate change if corpus PDFs with meaningful tables appear.
