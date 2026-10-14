# RAG QA System: user flows

## User flows

### Flow 1: Upload and ingest
1. The user sends a file with optional metadata.
2. The API validates type and size, computes a content hash, stores the raw file, creates a document record, and enqueues an ingestion job. It responds with the document id and status `queued`.
3. The worker parses the file into structured text (sections, pages, tables as text).
4. The worker chunks the text with the collection's chunking strategy and attaches metadata to each chunk (document id, version, section, page, position).
5. The worker embeds the chunks and writes them to the vector store. In parallel it updates the metadata DB.
6. Status becomes `ready`. On any failure, status becomes `failed` with a reason, and partial vectors are cleaned up.

Edge cases: duplicate hash, empty or unparseable file, oversized file, embedding service down (retry with backoff).

### Flow 2: Upload a new version
1. The user selects an existing document and uploads a file.
2. The system compares the content hash with the latest version. If equal, it reports "no changes" and stops.
3. Otherwise it creates version N+1 and runs Flow 1 for it.
4. When ingestion succeeds, the `latest` pointer moves to N+1. Until then, queries keep using version N.
5. Old versions stay searchable only when explicitly requested.

### Flow 3: Ask a question (default pipeline)
1. The user submits a question, optionally with filters (documents, tags, version) and a pipeline config.
2. Optional pre-retrieval steps run in a configured order: query rewriting, Self-Query (extract filters), HyDE (generate hypothetical document). If Self-Query resolves a document name to a tombstone, the pipeline stops here and responds that the document was deleted on its deletion date (Flow 7).
3. Retrieval: embed the query (or the hypothetical document), search the vector store with filters, optionally merge with BM25, optionally rerank.
4. Context assembly: select top chunks, expand with neighbors or parent chunk if the strategy requires it, fit into the token budget.
5. Generation: the LLM answers only from the context and returns citations. If the context is insufficient, it says so.
6. The response contains the answer, citations, and a trace id.

### Flow 4: Follow-up question
1. The user asks a follow-up in an existing conversation.
2. Query rewriting uses the history to produce a standalone question.
3. The rest follows Flow 3.

### Flow 5: Inspect a trace
1. The Experimenter opens the trace for a request.
2. They see each stage with inputs, outputs, scores, and latency, and can find which stage lost the relevant chunk (for example: Self-Query produced a wrong filter, or HyDE drifted).

### Flow 6: Evaluate pipeline configs
1. The Experimenter picks a golden set and one or more pipeline configs.
2. The runner executes every question for each config, over several seeds where generation is involved.
3. The system computes retrieval metrics and answer metrics and writes a report with the baseline first.
4. The Experimenter compares reports and decides what to keep.

### Flow 7: Delete a document
1. The user requests deletion of a document.
2. The system marks the document as deleted in the registry (deletion date set). From this moment the document is hidden from lists and its chunks are excluded from retrieval.
3. The system deletes the document's vectors from every index, then its raw files.
4. The registry keeps a tombstone (document id, title, deletion date) and removes versions and metadata. Each step is safe to retry.
5. Later, a question that names the deleted document gets a response saying it was deleted on that date (story A6, Flow 3 step 2).

Edge cases: deletion fails halfway (retry resumes from the marked-deleted state), a new upload with the same title (a new document, the tombstone stays), a version upload racing with deletion (rejected for a deleted document).
