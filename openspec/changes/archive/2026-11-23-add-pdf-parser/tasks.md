# Tasks

## 1. Decision record

- [x] 1.1 Write ADR-0024 in `docs/ADR.md` per design.md (Status/Context/Decision/Consequences, amends ADR-0001, dated 2026-11-23); verify it matches the existing ADR format

## 2. PDF parser

- [x] 2.1 Add `pymupdf` to `pyproject.toml` and run `uv sync --extra dev`; verify `uv run python -c "import pymupdf"` succeeds
- [x] 2.2 Implement `PdfParser` in `src/ragqa/ingestion/parsers/pdf_parser.py` (text extraction, best-effort sections, Info metadata, permanent errors for scanned/empty/encrypted) and register and export it in `parsers/__init__.py`; verify `default_registry().for_filename("x.pdf")` returns `PdfParser`
- [x] 2.3 Add `tests/unit/test_pdf_parser.py` building PDFs in memory with PyMuPDF, covering text extraction and normalization, heading sections, flat-section fallback, scanned/empty rejection, encrypted rejection, metadata mapping, and span invariants via `check_chunks`; verify `uv run pytest tests/unit/test_pdf_parser.py` passes

## 3. Integration verification

- [x] 3.1 Run `make check` (ruff, ruff format --check, mypy, pytest); verify it is clean

## Workflow follow-up

- Archive the change to sync the MODIFIED supported-formats requirement and the ADDED PDF requirements into `openspec/specs/document-model/`.
- Verify the result with `openspec show document-model --type spec`.
