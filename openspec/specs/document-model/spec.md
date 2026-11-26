# document-model Specification

## Purpose
Defines how uploaded documents are parsed into one normalized text with spans, and which formats are supported through an open parser registry.

## Requirements

### Requirement: Supported document formats
The system SHALL accept `.md`, `.docx` and `.pdf` uploads and reject any other format with a clear error. (ADR-0001, ADR-0024)

#### Scenario: Supported format accepted
- **WHEN** a user uploads a `.md`, `.docx` or `.pdf` file
- **THEN** the upload is accepted for ingestion

#### Scenario: Unsupported format rejected
- **WHEN** a user uploads a TXT or HTML file
- **THEN** the upload is rejected with a clear error naming the unsupported format

### Requirement: Normalized document text
A parser SHALL return one normalized text per document, in NFC with `\n` line endings. (ADR-0021)

#### Scenario: Parsing normalizes the text
- **WHEN** a document is parsed
- **THEN** its text is NFC-normalized and uses `\n` line endings

### Requirement: Spans into the normalized text
The parser SHALL expose spans for sections and code blocks into the one normalized text, and each chunk SHALL keep its source span so evidence matching and citations use the same character offsets. (ADR-0021)

#### Scenario: Chunk span indexes the normalized text
- **WHEN** a chunk is produced
- **THEN** its source span indexes into the same normalized text the parser returned

#### Scenario: Evidence and citations share offsets
- **WHEN** evidence is matched or a citation is resolved
- **THEN** both use character offsets into the one normalized text

### Requirement: Open parser registry
The source format SHALL be an open string key resolved by a parser registry; adding a format means adding a parser, not editing the document model. The API SHALL reject an upload whose extension has no parser. (ADR-0021)

#### Scenario: New format added through a parser
- **WHEN** a parser declaring a new format key and extension is registered
- **THEN** that format is accepted without changing the core document types

#### Scenario: Extension without a parser rejected
- **WHEN** an upload's extension has no registered parser
- **THEN** the API rejects the upload

### Requirement: Code block marking
The parser SHALL mark code blocks as code spans; chunks SHALL carry `has_code`, and chunkers can avoid splitting a code block. (ADR-0021)

#### Scenario: Fenced code is marked and kept intact
- **WHEN** a markdown document with fenced code is parsed and chunked
- **THEN** the code block is marked as a code span and a chunker can avoid splitting it

### Requirement: PDF extraction scope
The PDF parser SHALL extract text from text-based PDFs and SHALL reject a PDF with no extractable text (such as a scanned image-only PDF) or one that is password-protected as a permanent error. (ADR-0024)

#### Scenario: Text-based PDF parsed
- **WHEN** a text-based PDF is uploaded
- **THEN** its text is extracted into a normalized ParsedDocument

#### Scenario: Scanned PDF rejected
- **WHEN** a PDF has no extractable text
- **THEN** ingestion fails permanently with a clear error

#### Scenario: Password-protected PDF rejected
- **WHEN** a PDF is password-protected
- **THEN** ingestion fails permanently with a clear error

### Requirement: PDF section inference
The PDF parser SHALL produce sections for detectable headings and fall back to a single section covering the document when no headings are detectable; section detection is best-effort. (ADR-0024)

#### Scenario: Headings detected
- **WHEN** a PDF has visually distinct headings
- **THEN** sections are produced with heading paths into the normalized text

#### Scenario: No detectable headings
- **WHEN** a PDF has no detectable headings
- **THEN** a single section covers the document text
