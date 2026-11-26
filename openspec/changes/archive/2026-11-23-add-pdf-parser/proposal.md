# Proposal

## Why

PDF was excluded from v1 (ADR-0001) because layout extraction is unreliable, but ADR-0021's open parser registry was designed so a format can be added without changing the domain model or the API. Users need to ingest PDF documents. This adds a text-based PDF parser and reverses ADR-0001's scope through a new decision record, ADR-0024.

## What Changes

- Add a `PdfParser` built on PyMuPDF and register it for `.pdf` in `default_registry()`.
- Extract text from text-based PDFs into the normalized `ParsedDocument` contract (NFC, `\n`, spans), infer sections best-effort from font size and bold, and map the PDF Info dictionary to `DocumentMetadata`.
- Reject scanned or image-only PDFs (no extractable text) and password-protected PDFs as permanent errors.
- Add the `pymupdf` dependency (AGPL-3.0, accepted by the owner).
- Write ADR-0024 amending ADR-0001, and modify the `document-model` supported-formats requirement.
- No change to the frozen evaluation protocol (ADR-0019) or the golden set; the corpus stays Kubernetes markdown.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `document-model`: supported formats now include `.pdf`; add requirements for PDF extraction scope (reject scanned and encrypted PDFs) and best-effort section inference.

## Impact

- New `src/ragqa/ingestion/parsers/pdf_parser.py`; register and export it in `src/ragqa/ingestion/parsers/__init__.py`.
- `pyproject.toml`: add `pymupdf`. New unit tests in `tests/unit/test_pdf_parser.py`.
- `docs/ADR.md`: add ADR-0024.
- `openspec/specs/document-model/spec.md` updated on archive.
- The API accepts `.pdf` uploads automatically through the registry; no API code change.
- Adds an AGPL-3.0 licensed dependency.
