"""Prompt texts of the query pipeline. `PROMPT_VERSION` is logged with every evaluation run
(ML_DESIGN section 6): change it whenever any text here changes in a way that can move results."""

from collections.abc import Sequence

from ragqa.domain.retrieval import MetadataFilter, RetrievedChunk

PROMPT_VERSION = "2"

# The generator is told to answer with exactly this sentence when the context is not enough,
# so abstention can be detected without a judge (ML_DESIGN, abstention metric).
REFUSAL = "I cannot answer this from the available documents."

REWRITE_SYSTEM = """You turn the user's latest question into one standalone search query for a \
documentation search engine.
- Use the earlier conversation only to resolve references such as "it", "that" or "the second one".
- Keep product names, flags, field names and identifiers exactly as written.
- If the question is already standalone, return it unchanged.
- Output only the query, with no quotes and no explanation."""

SELF_QUERY_SYSTEM = """You split a question about documents into a semantic query and metadata \
filters.
Fill a filter only when the question names that value explicitly. Never guess or infer one.
- semantic_query: the question without the parts that only express filters. Keep every other word.
- doc_type: the kind of page asked for, for example concept, task, tutorial, reference.
- tags: topic labels named in the question.
- author: a person named as the author of the document.
- language: a language named as the language of the document.
- date_from, date_to: ISO dates (YYYY-MM-DD) when the question limits the document date.
- document_title: the title of one specific document the question refers to by name.
Use null (or an empty list) for everything that is not named."""

HYDE_SYSTEM = """You write a short passage of technical documentation that would answer the \
user's question, in the style of the documentation itself.
- Three to five sentences, plain text, no headings and no lists.
- Write it as if it were true. Do not mention uncertainty or that you were asked a question."""

GROUNDED_SYSTEM = f"""You answer questions about documentation using only the numbered \
excerpts you are given. The filters below were applied when selecting the excerpts.
- Cite the excerpts that support each statement with their numbers in square brackets, \
for example [1] or [2][3].
- Use only facts that the excerpts state. Do not use outside knowledge.
- If the excerpts do not contain the answer, reply with exactly: {REFUSAL}
- Be direct and concise. Keep commands, flags and identifiers exactly as written."""

CLOSED_BOOK_SYSTEM = """You answer questions about documentation from your own knowledge, \
without any excerpts. Be direct and concise.
Keep commands, flags and identifiers exactly as written."""


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        path = " > ".join(chunk.section_path)
        head = f"[{number}] {chunk.title}" + (f" > {path}" if path else "")
        blocks.append(f"{head} (version {chunk.version})\n{chunk.text}")
    return "\n\n".join(blocks)


def grounded_user_message(
    question: str, resolved_filters: MetadataFilter, chunks: Sequence[RetrievedChunk]
) -> str:
    notnull_filters = {k: v for k, v in resolved_filters.model_dump().items() if v}
    return (
        f"Excerpts:\n\n{format_context(chunks)}\n\n"
        f"Filters: {notnull_filters if notnull_filters else 'No filters were applied'}\n\n"
        f"Question: {question}"
    )
