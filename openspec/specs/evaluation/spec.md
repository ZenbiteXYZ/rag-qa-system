# evaluation Specification

## Purpose
Defines the hand-written golden set, the Kubernetes evaluation corpus, and the frozen protocol v1 that governs how results are produced, measured and compared.

## Requirements

### Requirement: Hand-written golden set
The golden question set SHALL be written by hand and stored under version control, some with metadata cues; synthetic questions are a supplement only, and differences between configs are reported with variance across runs, not point estimates. (ADR-0005)

#### Scenario: Golden set is version-controlled
- **WHEN** the golden set is changed
- **THEN** the change is stored under version control

### Requirement: Golden set shape
The golden set SHALL contain 6 questions in each of 7 categories (42 total), split 3 dev and 3 test per category with a fixed seed. Evidence SHALL be verbatim quotes that a validator resolves against the parser output. (ADR-0019, ADR-0005)

#### Scenario: Split is 3 dev and 3 test per category
- **WHEN** the golden set is validated
- **THEN** each of the 7 categories has 3 dev and 3 test questions, 42 in total

#### Scenario: Evidence quotes resolved
- **WHEN** a question's evidence is validated
- **THEN** each verbatim quote is resolved against the parser output

### Requirement: Evaluation corpus
The corpus SHALL be kubernetes/website content/en/docs (CC BY 4.0), releases 1.32, 1.34 and 1.36, a subset such as concepts and tasks, English only. Hugo shortcodes are rendered to plain Markdown per release, with source commit and per-file hash in a manifest, kept outside the repo read-only. Attribution is required; absent author metadata is synthetic from a versioned mapping. (ADR-0006, ADR-0007)

#### Scenario: Corpus prepared per release with a manifest
- **WHEN** the corpus is prepared
- **THEN** shortcodes are rendered per release and the source commit and per-file hashes are recorded in a manifest

#### Scenario: Absent author filled from a mapping
- **WHEN** a corpus page has no reliable author
- **THEN** author metadata is assigned from a versioned mapping file

### Requirement: Retrieval metrics
Retrieval SHALL be measured as evidence coverage by the top-k chunks with k=3 and threshold 0.8, with sensitivity reports at thresholds 0.5 and 1.0 and at k=5. Multi-evidence questions average over their required items; all metrics are per question and averaged over questions. (ADR-0019)

#### Scenario: Coverage at 3 computed
- **WHEN** retrieval is evaluated
- **THEN** evidence coverage by the top-3 chunks at threshold 0.8 is computed per question and averaged

### Requirement: Levels and comparison
The protocol SHALL report retrieval (Coverage@3), filter extraction (exact match), end-to-end (answer correctness) and system (p95 latency, tokens per request). Comparison uses a dense fixed-size baseline, one change at a time, a closed-book control, 2 repetitions, and paired per-question comparison with an interval over questions. (ADR-0019)

#### Scenario: Paired comparison against the baseline
- **WHEN** a configuration is compared
- **THEN** it is compared one change at a time against the dense baseline, paired per question with an interval over questions

### Requirement: Immutable run records
Every run SHALL store its config, versions, raw per-question outputs and judge outputs as immutable artifacts, and metrics SHALL be computed from them only, so a changed metric can be recomputed without regenerating answers. (ADR-0019)

#### Scenario: Metric recomputed from stored outputs
- **WHEN** a metric definition is recomputed
- **THEN** it is computed from the stored raw outputs without regenerating answers

### Requirement: Pinned models and judge gate
The protocol SHALL pin generator anthropic/claude-haiku-5.5-20261007, judge openai/gpt-6-luna-20260922 (reasoning effort low, structured output, seed set), and embedder Snowflake/snowflake-arctic-embed-m-v2.0 at revision 95c2741480856aa9666782eb4afe11959938017f. The judge SHALL pass a calibration gate against 30 hand-labeled dev answers before its scores are used. (ADR-0019)

#### Scenario: Uncalibrated judge refused
- **WHEN** the judge has not passed the calibration gate
- **THEN** its scores are not used unless explicitly overridden

### Requirement: Evaluation protocol v1 is frozen
Protocol v1 SHALL be treated as fixed. Any change to metrics, splits, the golden set, the judge or its rubric SHALL create protocol version 2 with a new ADR, and results from different protocol versions SHALL NOT be mixed. (ADR-0019)

#### Scenario: A metric definition changes
- **WHEN** a change alters a metric, split, golden question, judge or rubric
- **THEN** it is recorded as protocol v2 with a new ADR, not edited into v1
