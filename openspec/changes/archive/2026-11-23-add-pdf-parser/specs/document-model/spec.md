# Spec Delta

## MODIFIED Requirements

### Requirement: Supported document formats
The system SHALL accept `.md`, `.docx` and `.pdf` uploads and reject any other format with a clear error. (ADR-0001, ADR-0024)

#### Scenario: Supported format accepted
- **WHEN** a user uploads a `.md`, `.docx` or `.pdf` file
- **THEN** the upload is accepted for ingestion

#### Scenario: Unsupported format rejected
- **WHEN** a user uploads a TXT or HTML file
- **THEN** the upload is rejected with a clear error naming the unsupported format

## ADDED Requirements

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
