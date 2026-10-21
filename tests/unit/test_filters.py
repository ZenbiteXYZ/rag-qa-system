import uuid
from datetime import UTC, datetime

from ragqa.domain.pipeline import SelfQueryConfig
from ragqa.domain.retrieval import KnownDocument
from ragqa.query.filters import normalize, resolve_document, resolve_value, similarity

CONFIG = SelfQueryConfig()
AUTHORS = ("I. Petrov", "A. Ivanova", "M. Petrenko")


def test_normalize_ignores_case_and_punctuation() -> None:
    assert normalize("  I.  PETROV ") == "i petrov"


def test_surname_is_a_whole_word_match() -> None:
    assert similarity("Petrov", "I. Petrov") >= 0.9
    assert similarity("Pet", "I. Petrov") < 0.9


def test_exact_value_ignores_case() -> None:
    result = resolve_value("i. petrov", AUTHORS, CONFIG)
    assert (result.outcome, result.resolved) == ("exact", "I. Petrov")


def test_a_clearly_best_match_is_substituted() -> None:
    result = resolve_value("Petrov", AUTHORS, CONFIG)
    assert (result.outcome, result.resolved) == ("substituted", "I. Petrov")


def test_two_close_candidates_are_asked_about() -> None:
    result = resolve_value("Petro", AUTHORS, CONFIG)
    assert result.outcome == "asked"
    assert set(result.candidates) == {"I. Petrov", "M. Petrenko"}


def test_nothing_close_is_unresolved() -> None:
    assert resolve_value("Zhang", AUTHORS, CONFIG).outcome == "unresolved"
    assert resolve_value("anyone", (), CONFIG).outcome == "unresolved"


def test_thresholds_come_from_the_config() -> None:
    strict = SelfQueryConfig(match_threshold=1.0, candidate_floor=0.8)
    assert resolve_value("Petrov", AUTHORS, strict).outcome == "asked"


def _doc(title: str, deleted: bool = False) -> KnownDocument:
    return KnownDocument(uuid.uuid4(), title, datetime.now(UTC) if deleted else None)


def test_document_resolution_live_deleted_and_asked() -> None:
    live, gone = _doc("Drain a Node"), _doc("Old Guide", deleted=True)
    docs = (live, gone)

    found = resolve_document("drain a node", docs, CONFIG)
    assert found.outcome == "exact" and found.live == (live,) and found.deleted is None

    deleted = resolve_document("Old Guide", docs, CONFIG)
    assert deleted.deleted == gone and deleted.live == ()

    unknown = resolve_document("Quantum Networking", docs, CONFIG)
    assert unknown.outcome == "unresolved"


def test_a_live_document_wins_over_a_tombstone_with_the_same_title() -> None:
    live, gone = _doc("Guide"), _doc("Guide", deleted=True)

    found = resolve_document("Guide", (gone, live), CONFIG)

    assert found.live == (live,) and found.deleted is None
