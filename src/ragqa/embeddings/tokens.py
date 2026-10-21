"""Token counting with the embedder's own tokenizer, so chunk sizes mean what the model sees
(ADR-0012). The tokenizer file comes from the Hugging Face Hub at the pinned revision (ADR-0019)
and is cached on disk after the first download."""

from functools import lru_cache

from tokenizers import Tokenizer

from ragqa.domain.indexes import Index
from ragqa.ingestion.chunkers import TokenCounter, TokenCounterFactory
from ragqa.settings import Settings


class HfTokenCounter:
    def __init__(self, tokenizer: Tokenizer) -> None:
        # The counter must see the whole text: a stored truncation limit would cap every count.
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self._tokenizer = tokenizer

    def count(self, text: str) -> int:
        # Without the special tokens the model adds around a passage, which are not chunk text.
        return len(self._tokenizer.encode(text, add_special_tokens=False).ids)


@lru_cache(maxsize=4)
def _load(model: str, revision: str) -> HfTokenCounter:
    return HfTokenCounter(Tokenizer.from_pretrained(model, revision=revision))


def hf_token_counter_factory(settings: Settings) -> TokenCounterFactory:
    def factory(index: Index) -> TokenCounter:
        if index.dense_model != settings.embedder_model:
            raise ValueError(
                f"no tokenizer for {index.dense_model!r}: this deployment embeds with "
                f"{settings.embedder_model!r}"
            )
        # Downloads on first use, so call it from a worker thread.
        return _load(settings.embedder_model, settings.embedder_revision)

    return factory
