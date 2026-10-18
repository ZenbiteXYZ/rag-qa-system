# ML system design and evaluation plan

Status: protocol v1, recorded in ADR-0019 and written from the decisions in [ADR.md](ADR.md). Nothing here is a result: no experiment has been run yet.

## 1. Goal and scope

Build a retrieval-augmented QA service over versioned documents and measure which retrieval techniques actually improve it. The system is a lab, so the evaluation protocol matters as much as the service.

The questions this document answers:
- What do we expect each technique to change, and why? (section 2)
- What data do we evaluate on, and how do we keep it clean? (section 3)
- How do we measure, compare and report? (sections 4 and 5)
- What can make the results wrong? (section 7)

Out of scope: fine-tuning any model. Models are used as they are (ADR-0009).

## 2. Techniques under test and hypotheses

Every technique is a config switch (ADR-0010) and is compared with the baseline in section 4. Hypotheses are written before running anything, so a result can contradict them.

| Technique | Config | Hypothesis | Could hurt when | Questions that test it |
|-----------|--------|-----------|-----------------|------------------------|
| Chunking: fixed size with overlap | index | Baseline chunker | Splits sections mid-thought | all |
| Chunking: structure-aware (headings) | index | Better for questions answered inside one section | Very long or very short sections | single-section, multi-section |
| Chunking: semantic | index | Possible gain on prose-heavy pages, unclear on technical docs | Costs more to build, may split lists and code | single-section |
| Hybrid (dense + BM25, RRF) | retrieve | Helps questions with exact tokens (flags, field names, kinds) | Little or no gain on paraphrased questions | exact-token |
| Query rewriting | rewrite | Helps follow-up and underspecified questions | Rewrite drifts from the user's intent | follow-up |
| HyDE | hyde | Helps short or abstract questions | Hypothetical text invents wrong details that pull retrieval off | abstract, single-section |
| Self-Query | self_query | Helps questions with metadata cues, when the extracted filter is correct | Wrong filter removes the right chunk | metadata-cue |
| Reranking | rerank | Improves the order of the top candidates | Extra latency, no gain if the candidates are already good | multi-section |
| Embedding truncation (MRL 256) | index | Small quality loss for much smaller vectors (the model card reports about one BEIR point) | Quality loss on our corpus is larger than reported | all |

Versioning is also behavior to test: a question whose answer differs between two releases must be answered from the requested version, and from the latest one by default.

## 3. Data

### Corpus
- Kubernetes docs, `content/en/docs` subset (`concepts/`, `tasks/`), taken from 2 or 3 release branches (ADR-0006). Licence CC BY 4.0, so attribution is needed in any public report.
- Counts of documents, chunks and tokens per version are recorded in the report of the first ingestion.
- The corpus is public and almost certainly seen by the LLMs during pretraining. See "closed-book control" in section 7.

### Metadata
- `doc_type` comes from the front matter `content_type`.
- `author` is synthetic where the corpus has none (ADR-0007). The mapping file is versioned in git and generated with a fixed seed. Author-filter questions therefore test the filter mechanism, not natural cues in the text.

### Golden set
Written by hand (ADR-0005), 42 questions, stored in `eval/golden/` and versioned in git.

| Category | Count | Purpose |
|----------|-------|---------|
| Single-section factoid | 6 | Basic retrieval and grounding |
| Multi-section | 6 | Needs more than one chunk |
| Exact-token | 6 | Flags, field names, identifiers (hybrid) |
| Metadata-cue | 6 | Self-Query and filter validation |
| Version-sensitive | 6 | Answer differs between releases |
| Follow-up | 6 | Query rewriting, needs conversation history |
| Unanswerable | 6 | The system must say it cannot answer |

Uniform proportions: 6 questions per category, 42 in total, split 3 dev and 3 test per category (21 and 21). This is a pilot-scale set: it can show large differences only, so results are exploratory. If they are inconclusive, the set is extended under protocol v2 with a new ADR.

Record fields per question (draft): `id`, `category`, `question`, `history` (for follow-ups), `expected_filters`, `evidence` (the tagged quotes, see below), `reference_answer`, `split`.

Two problems to settle before writing questions:
1. **Ground truth that survives re-chunking.** Each index chunks differently, so chunk ids from one index mean nothing in another. Decision: the relevant evidence is tagged in the golden set independently of any index, and both retrieval ("was the evidence found") and the answer ("was it used") are compared against the tags. Accepted by the author:
   - **Tag format:** each evidence item has document, version, section path, and a verbatim quote, with a flag for required or alternative. The quote is the minimal passage that is sufficient to answer (one to three sentences). A validator resolves each quote against the parser output of that version and requires exactly one match, after normalizing line endings, Unicode form and whitespace. Offsets are derived by the validator and never written by hand. The validator runs whenever the parser changes.
   - **Chunks** keep their source span (character offsets in the parser output) separately from their text.
   - **Validator:** `ragqa.evaluation.evidence`, run as `python -m ragqa.evaluation.evidence eval/golden --text-source module:function`. Golden files are YAML lists in `eval/golden/`. The text source returns the parser output for a (document, version) and is wired up once the parser exists. The parser output must be NFC-normalized.
   - **Matching by character overlap.** `coverage@k` is the share of the required evidence characters covered by the union of the top-k chunks (overlapping intervals are merged first). It is computed on the retrieved candidates and, separately, on the final context after neighbor expansion.
   - **Retrieval metrics, k and the threshold** are defined in section 4.
   - **Multi-evidence questions:** the coverage of a question is the average of the per-item coverage over its required items. Items with `required: false` are not counted.
2. **Dev and test split.** 42 questions, split 50/50 (21 and 21). Prompts, top-k and thresholds are tuned on the dev part only, and ablations are compared on dev. The test part is used once per final comparison (baseline against the chosen configuration), not to choose between all the ablations. Proposal: stratify the split by category with a fixed seed, and write the questions before running experiments. At 21 test questions a score near 0.7 has a standard error of about 0.10, and a paired difference between two configurations is distinguishable only if it is about 0.13 or larger (assuming a per-question standard deviation of 0.3, to be replaced by the measured value). Per-category numbers (3 questions each) are descriptive only.

## 4. Evaluation protocol

### Levels
1. **Retrieval only** (no generation): is the right evidence in the retrieved set, and how high is it ranked? Cheap, deterministic, run for every index and retrieval config.
2. **Filter extraction** (Self-Query): are the extracted filters the expected ones?
3. **End to end**: is the answer correct and grounded, are the citations valid, does the system abstain when it should?
4. **System**: latency per stage and tokens per request.

### Baseline
Fixed-size chunker, dense-only retrieval, no rewriting, no HyDE, no Self-Query, no reranking, same LLM and prompts as the variants. Every comparison is against this configuration, one change at a time (ablation), then the best combination at the end.

### Metrics
Unless stated otherwise, a metric is computed per question and then averaged over questions. The question is the unit, not chunks, filters or claims. Category breakdowns are descriptive only (3 test questions per category).

Metrics are either deterministic (computed from the evidence tags, filters and citations) or LLM-judged. LLM-judged metrics use the judge pinned in section 6 (a different model family from the generator), a versioned rubric prompt and structured JSON output. Before its scores are used, the judge has to pass the calibration gate below. Abstention and hallucination are reported as separate rates and are not combined into one cost score.

#### Retrieval
Deterministic, computed against the evidence tags with character overlap. Parameters fixed before running: k = 3 and threshold = 0.8. Sensitivity reports at thresholds 0.5 and 1.0, and at k = 5 because multi-evidence questions need more than one chunk.

| Metric | Meaning |
| ------ | ------- |
| Coverage@k | Share of the required evidence characters covered by the union of the top-k retrieved chunks (k = 3), averaged over answerable questions. Continuous from 0 to 1. Primary retrieval metric. Computed on the ranked candidates and again on the final context after neighbor expansion. |
| MRR | Mean over answerable questions of 1 / r, where r is the smallest number of top-ranked chunks whose cumulative coverage reaches the threshold (0.8), or 0 if it never does. Rewards finding the evidence early. |
| Recall@k | Share of answerable questions whose Coverage@k reaches the threshold (0.8). Pass or fail per question. Sensitivity reports at thresholds 0.5 and 1.0 and at k = 5. |
| Precision@k | Share of the characters in the top-k chunks that belong to the evidence. Stops very large chunks from scoring well on coverage alone. Diagnostic. |
| Containment and overlap | Share of questions where a single chunk contains the whole evidence, and share where any chunk overlaps it. With Recall@k they bracket it from below and above, and the gap shows how often the chunker splits evidence. Diagnostic. |

#### Filter extraction
Measured on all questions, not only the metadata-cue ones, so spurious filters are counted. A filter is a (field, value) pair, and values are compared after the validation step of ADR-0008.

| Metric | Meaning |
| ------ | ------- |
| Precision | Of the filters the system applied, the share that are in the expected filters. A wrongly extracted filter lowers it, and so does any filter on a question that expects none. |
| Recall | Of the expected filters, the share the system applied. A missed filter lowers it. |
| F1-score | Harmonic mean of precision and recall, computed per question and then averaged. A question with no expected and no applied filters scores 1 on all three. |
| Exact match | Share of questions whose applied filter set equals the expected set. Closest to what the user sees, since one wrong filter can hide the right chunk. |
| Substitution outcome | How often an unknown value was auto-corrected, asked about, or left unresolved, and whether the auto-corrections were right. Measures the fuzzy-match behavior of ADR-0008. |

#### End to end
Faithfulness and Answer Relevancy follow RAGAS ideas. If you use that library, check its definitions for the version you pin, because they have changed between versions.

| Metric | Meaning |
| ------ | ------- |
| Faithfulness | Share of the claims in the answer that are supported by the final context. LLM judge. Catches hallucination, but an answer can be faithful to the context and still wrong. Diagnostic. |
| Answer Relevancy | How directly the answer addresses the question, so off-topic, padded or incomplete answers score low. It does not use the reference answer. LLM judge. Diagnostic. |
| Answer correctness | Whether the answer agrees with the reference answer: facts match, nothing contradicts it, nothing essential is missing. LLM judge using a G-Eval style rubric. Primary end-to-end metric. |
| Citation validity | Share of citations that point to a chunk that is in the final context. Deterministic. Catches invented citations. |
| Citation support | Share of required evidence covered by the chunks the answer cites. This is the "used in the answer" side of the tags. Deterministic. |
| Abstention | On unanswerable questions, the share where the system declines (abstention recall). On answerable questions, the share where it wrongly declines (false abstention rate). Report both. |
| Version accuracy | On version-sensitive questions, the share of answers whose cited evidence comes from the expected version. Deterministic. |

#### System
Reported per stage and in total, with p50 and p95 for timings.

| Metric | Meaning |
| ------ | ------- |
| Latency/Stage | Wall-clock time of each pipeline stage per request. Prices each technique, because rewriting and HyDE each add an LLM call. |
| Tokens/Request | Prompt and completion tokens per request and per stage. With provider prices it gives the cost per request. |
| Memory Usage | Peak memory of each service during a run. Recorded for sizing (embedder, Qdrant, worker) and not used to compare techniques. Collection is deferred. |
| CPU usage | CPU use of each service during a run. Recorded for sizing and not used to compare techniques. Collection is deferred. |
| LLM calls/Request | Number of LLM calls per request. The simplest measure of what each technique adds. |
| Index size and build time | Chunks, vectors and disk size per index, and ingestion time. Needed to compare chunkers and the 256-dimension truncation. |

#### Primary metric per level
Decisions between configs use one metric per level, fixed before running. Everything else is a diagnostic.

| Level | Primary |
| ----- | ------- |
| Retrieval | Coverage@3 |
| Filter extraction | Exact match |
| End to end | Answer correctness |
| System | p95 end-to-end latency and tokens per request |

### Judge calibration gate
Before the first judged result is used:
1. Hand-label 30 answers from the dev half (21 questions, so take answers from at least two configurations or both repetitions) as correct, partly correct or wrong. Include answers that are fluent but wrong, and correct answers worded differently from the reference.
2. Run the judge on them twice. Report its agreement with your labels and the agreement between its own two runs.
3. The gate, fixed before looking at the result: exact agreement with your labels of at least 80%, and at most 10% of the answers you labeled wrong judged correct. With 30 labels this is a coarse check. If it fails, fix the rubric first. A new judge or a new rubric then counts as a protocol change.
4. Repeat the check whenever the judge, the rubric or the generator changes.

### Repetition and uncertainty
- LLM calls are not deterministic (the chosen models do not accept sampling settings, see section 6), so end-to-end runs are repeated. Each configuration is run twice. Run-to-run noise is estimated from the difference between the two runs of each question.
- Compare configs per question (the same questions in both), and report the number of questions that got better, worse and unchanged next to the averages.
- With so few questions, report an interval (for example a bootstrap over questions), and do not call a difference real if it lies within the run-to-run noise.
- Always report the baseline next to every result.

### Protocol changes
The evaluation protocol is versioned (`protocol_version` in every report). This document describes version 1. A change to metrics, splits, the golden set, the judge or its rubric creates a new version, is recorded as an ADR, and old results are not mixed with new ones.

## 5. Experiment plan

Indexing choices first, because each one needs a new index build (ADR-0013), then query-side techniques on the best index.

| Run | Change from baseline | Level |
|-----|---------------------|-------|
| E0 | Baseline | 1 to 4 |
| E0b | Closed-book control: no retrieval, same LLM | 3 |
| E1 | Structure-aware chunker | 1, 3 |
| E2 | Semantic chunker | 1, 3 |
| E3 | Hybrid retrieval | 1, 3 |
| E4 | Query rewriting | 1, 3 |
| E5 | HyDE (dense branch only, see ADR-0018) | 1, 3 |
| E6 | Self-Query | 1, 2, 3 |
| E7 | Reranking | 1, 3 |
| E8 | Best combination | 1 to 4 |
| E9 | Optional: embedding truncation to 256 dimensions | 1 |

Each run is an MLflow run (ADR-0015).

## 6. Reproducibility and tracking

Logged in every run: config hash, git sha, `protocol_version`, golden set version, model names and revisions (embedder with its Hugging Face revision, LLM alias and the provider model behind it), Qdrant and TEI image tags, prompt versions, repetition index, seeds. Library versions come from `uv.lock`.

### Models and pins
Identifiers are OpenRouter canonical slugs as listed on 2026-10-18. Both LLMs are called through the LiteLLM proxy (ADR-0016). Log the provider that served each call if the response reports it, because OpenRouter can route to different providers.

| Role | Model | Pinned identifier | Settings and notes |
|------|-------|-------------------|--------------------|
| Generator and pipeline stages | Claude Haiku 5.5 | `anthropic/claude-haiku-5.5-20261007` | Sampling settings such as temperature cannot be changed on this model. Reasoning effort is part of the pipeline config and is the same in all compared configurations (default `medium`). |
| Judge | GPT-6 Luna | `openai/gpt-6-luna-20260922` | Reasoning effort `low`, structured JSON output, `seed` set and logged. The listed parameters include `seed`, `reasoning_effort` and `structured_outputs` but no temperature. |
| Embedder | Snowflake Arctic Embed M v2.0 | `Snowflake/snowflake-arctic-embed-m-v2.0` at revision `95c2741480856aa9666782eb4afe11959938017f` | Served by TEI. Queries use the prefix `query: `. |

What is random: the dev/test split (fixed seed, stored in the golden set), the synthetic author mapping (fixed seed), and LLM sampling, which cannot be controlled: neither the generator nor the judge accepts a temperature setting through OpenRouter, so runs are repeated (section 4). The judge accepts a `seed`, which is set on a best-effort basis and logged.

## 7. Risks and failure modes

| Risk | Effect | Mitigation |
|------|--------|-----------|
| LLM memorized the Kubernetes docs | Correct answers without retrieval, so retrieval gains are hidden | Closed-book control E0b, version-sensitive and release-specific questions |
| Questions written while looking at specific chunks | Leak between the questions and one chunker | Write questions from the documents, not from chunks, and use chunker-independent ground truth |
| Tuning on the whole golden set | Overfitting to the golden set | Dev and test split, test used once per comparison |
| Small golden set | Wide uncertainty, false conclusions | Intervals, paired comparison, repeat runs, claim only large differences, extend under protocol v2 if inconclusive |
| LLM judge bias or drift | Scores move with the judge, not the system | Fixed judge model and prompt, check against own labels |
| Wrong Self-Query filter | The right chunk is filtered out | Filter validation (ADR-0008), filter-extraction metric |
| HyDE drift | Retrieval pulled toward invented content | Dense branch only, ablation E5 |
| Hallucinated citations | Answer cites evidence that was not retrieved | Citation check against the context, count invalid citations |
| Stale version served | Answer from the wrong release | Version-sensitive questions, `is_latest` invariants (ADR-0014) |
| Synthetic authors | Filters look better than real metadata would | Report filter results as mechanism tests only |
| Provider model changes | Results drift between runs | Log provider model and date, re-run the baseline with each comparison |
## 8. Latency and cost

Targets for end-to-end latency (p50 and p95) and cost per request: latency for one request depends on model and reasoning config (thinking enabled or disabled, effort) but in general one common question should be answered in ~4-10 seconds and cost around 0.0004-0.0008$ for a small model like Haiku 5.5 (estimated for the baseline from about 2,000 input and 400 output tokens for generation, at the prices of 2026-10-18 and before thinking tokens; about 600 input and 150 output tokens for each additional LLM stage). The latency range is the p50 target and p95 is reported without a target. Latency has not been measured: E0 checks the estimate, and cost is recomputed from measured tokens. Latency and tokens are traced per stage, so every technique has its cost reported next to its quality gain.

## 9. Decision log

All decisions below are part of protocol v1 and are recorded in ADR-0019.

- Judge: `openai/gpt-6-luna-20260922`, behind the calibration gate in section 4.
- Repetitions: 2 per configuration.
- Multi-evidence questions: average of per-item coverage over the required items.
- Retrieval parameters: k = 3 and threshold 0.8, with sensitivity reports.
