import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from ragqa.domain.documents import ParsedDocument, check_chunks
from ragqa.embeddings.tokens import HfTokenCounter
from ragqa.ingestion.chunkers import FixedSizeChunker, RecursiveChunker, default_registry
from ragqa.ingestion.chunkers.langchain_chunkers import _locate
from ragqa.ingestion.parsers import default_registry as parsers


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())


GUIDE = (
    """# Drain a node

Use kubectl drain to remove all pods from a node before maintenance.
The command cordons the node first, so nothing new is scheduled there.

## Options

```bash
kubectl drain my-node --ignore-daemonsets --delete-emptydir-data
```

Pods that are not managed by a controller block the drain unless you pass --force.
A second paragraph about timeouts and grace periods follows here, with enough words
to make the section longer than a single chunk of twenty words when it is split.

## Uncordon

Run kubectl uncordon to make the node schedulable again.
"""
    * 3
)


@pytest.fixture(scope="module")
def doc() -> ParsedDocument:
    return parsers().for_filename("guide.md").parse(GUIDE.encode(), filename="guide.md")


def chunkers() -> list[object]:
    return [
        FixedSizeChunker({"chunk_chars": 200, "overlap_chars": 40}, WordCounter()),
        RecursiveChunker({"chunk_tokens": 20, "overlap_tokens": 4}, WordCounter()),
    ]


@pytest.mark.parametrize("chunker", chunkers(), ids=lambda c: type(c).__name__)
def test_spans_match_the_text_and_pass_the_contract(
    chunker: FixedSizeChunker, doc: ParsedDocument
) -> None:
    chunks = chunker.chunk(doc)

    check_chunks(doc, chunks)
    assert len(chunks) > 3
    for chunk in chunks:
        assert doc.text[chunk.start : chunk.end] == chunk.text
        assert chunk.token_count == len(chunk.text.split())
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)


@pytest.mark.parametrize("chunker", chunkers(), ids=lambda c: type(c).__name__)
def test_every_word_of_the_document_is_in_some_chunk(
    chunker: FixedSizeChunker, doc: ParsedDocument
) -> None:
    covered = [False] * len(doc.text)
    for chunk in chunker.chunk(doc):
        covered[chunk.start : chunk.end] = [True] * (chunk.end - chunk.start)

    missing = [i for i, c in enumerate(covered) if not c and not doc.text[i].isspace()]
    assert missing == []


def test_neighbouring_chunks_overlap_when_asked(doc: ParsedDocument) -> None:
    chunks = FixedSizeChunker({"chunk_chars": 200, "overlap_chars": 60}, WordCounter()).chunk(doc)

    assert all(a.end > b.start for a, b in zip(chunks, chunks[1:], strict=False))
    no_overlap = FixedSizeChunker({"chunk_chars": 200, "overlap_chars": 0}, WordCounter()).chunk(
        doc
    )
    assert all(a.end <= b.start for a, b in zip(no_overlap, no_overlap[1:], strict=False))


def test_recursive_chunks_respect_the_token_limit(doc: ParsedDocument) -> None:
    chunks = RecursiveChunker({"chunk_tokens": 20, "overlap_tokens": 0}, WordCounter()).chunk(doc)

    assert max(c.token_count for c in chunks) <= 20


def test_section_path_and_code_flag(doc: ParsedDocument) -> None:
    chunks = FixedSizeChunker({"chunk_chars": 200, "overlap_chars": 0}, WordCounter()).chunk(doc)

    in_code = [c for c in chunks if "ignore-daemonsets" in c.text]
    assert in_code and all(c.has_code for c in in_code)
    assert not any(c.has_code for c in chunks if "uncordon to make" in c.text)
    first = chunks[0]
    assert first.section_path == ("Drain a node",)
    options = next(c for c in chunks if "Pods that are not managed" in c.text)
    assert options.section_path == ("Drain a node", "Options")


def test_repeated_text_gets_the_right_span() -> None:
    block = "same words here\n\n"
    doc = ParsedDocument(title="t", source_format="md", text=block * 6 + "the end")

    chunks = RecursiveChunker({"chunk_tokens": 3, "overlap_tokens": 0}, WordCounter()).chunk(doc)

    check_chunks(doc, chunks)
    # Six identical chunks must map to six different places, in order.
    assert [c.start for c in chunks] == [i * len(block) for i in range(6)] + [6 * len(block)]
    assert chunks[-1].text == "the end"


def test_locate_skips_a_repeat_inside_the_previous_chunk() -> None:
    # The previous chunk is [0, 7) "xy xy z". A short piece "xy" also appears inside it, but
    # the next chunk must end after the previous one.
    assert _locate("xy xy z xy", "xy", min_start=1, min_end=7) == 8
    with pytest.raises(ValueError, match="not a piece"):
        _locate("abc", "zzz", 0, 0)


@pytest.mark.parametrize(
    ("factory", "params"),
    [
        (FixedSizeChunker, {"chunk_chars": 100, "overlap_chars": 100}),
        (FixedSizeChunker, {"chunk_chars": 0}),
        (FixedSizeChunker, {"size": 100}),
        (RecursiveChunker, {"chunk_tokens": 10, "overlap_tokens": 10}),
        (RecursiveChunker, {"chunk_chars": 100}),
    ],
)
def test_bad_params_are_value_errors(
    factory: type[FixedSizeChunker], params: dict[str, int]
) -> None:
    with pytest.raises(ValueError):
        factory(params, WordCounter())


def test_empty_params_use_defaults(doc: ParsedDocument) -> None:
    for strategy in default_registry().strategies:
        chunker = default_registry().build(strategy, {}, WordCounter())
        check_chunks(doc, chunker.chunk(doc))


def test_default_registry_has_both_strategies() -> None:
    assert default_registry().strategies == ("fixed-size", "recursive")


def test_token_counter_ignores_a_stored_truncation_limit() -> None:
    vocab = {"[UNK]": 0, "one": 1, "two": 2}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.enable_truncation(max_length=2)

    assert HfTokenCounter(tokenizer).count("one two two one two") == 5
