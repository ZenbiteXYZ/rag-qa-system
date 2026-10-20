import unicodedata
from pathlib import Path

import pytest

from ragqa.evaluation.evidence import (
    Span,
    collapse_whitespace,
    load_provider,
    load_questions,
    main,
    resolve_quote,
    validate_questions,
)

TEXT = "Pods run containers.\nA Pod is the smallest\n  deployable unit.\r\nServices expose Pods."
TEXTS = {("doc.md", "v1"): TEXT}


def provider(document: str, version: str) -> str:
    return TEXTS[(document, version)]


def question(quote: str, **overrides: object) -> dict[str, object]:
    q: dict[str, object] = {
        "id": "q1",
        "category": "single_section",
        "evidence": [{"document": "doc.md", "version": "v1", "quote": quote}],
    }
    q.update(overrides)
    return q


def codes(questions: list[object]) -> list[str]:
    issues, _ = validate_questions(questions, provider)
    return [i.code for i in issues]


def test_collapse_whitespace_maps_back_to_original_offsets() -> None:
    flat, origin = collapse_whitespace("a  b\n\nc")

    assert flat == "a b c"
    # a, the first space of the run, b, the first newline of the run, c
    assert origin == [0, 1, 3, 4, 6]


def test_unique_quote_resolves_to_original_offsets() -> None:
    quote = "A Pod is the smallest deployable unit."

    resolution = resolve_quote(TEXT, quote)

    assert len(resolution.spans) == 1
    span = resolution.spans[0]
    assert TEXT[span.start : span.end].split() == quote.split()
    assert span == Span(TEXT.index("A Pod"), TEXT.index("unit.") + len("unit."))


def test_valid_question_has_no_issues_and_records_span() -> None:
    issues, spans = validate_questions([question("Services expose Pods.")], provider)

    assert issues == []
    assert spans[("q1", 0)].end == len(TEXT)


def test_ambiguous_quote_reports_every_match() -> None:
    issues, _ = validate_questions([question("Pod")], provider)

    assert [i.code for i in issues] == ["AMBIGUOUS"]
    assert "lengthen" in issues[0].message


def test_missing_quote_reports_where_it_diverges() -> None:
    issues, _ = validate_questions([question("A Pod is the largest unit")], provider)

    assert [i.code for i in issues] == ["NOT_FOUND"]
    assert "first 13 of 25 characters match" in issues[0].message


def test_quote_is_whitespace_insensitive_but_case_sensitive() -> None:
    assert codes([question("smallest   deployable\nunit")]) == []
    assert codes([question("a pod is the smallest")]) == ["NOT_FOUND"]


def test_unanswerable_must_not_have_evidence() -> None:
    assert codes([question("Pods run containers.", category="unanswerable")]) == [
        "UNEXPECTED_EVIDENCE"
    ]
    assert codes([{"id": "q2", "category": "unanswerable"}]) == []


def test_answerable_needs_evidence() -> None:
    assert codes([{"id": "q1", "category": "single_section"}]) == ["NO_EVIDENCE"]


def test_missing_field_unknown_document_and_duplicate_id() -> None:
    bad_field = {"id": "a", "category": "x", "evidence": [{"document": "doc.md"}]}
    unknown_doc = question("x", id="b")
    unknown_doc["evidence"] = [{"document": "nope.md", "version": "v1", "quote": "x"}]

    assert codes([bad_field]) == ["MISSING_FIELD"]
    assert codes([unknown_doc]) == ["TEXT_UNAVAILABLE"]
    assert codes([question("Pods run containers."), question("Pods run containers.")]) == [
        "DUPLICATE_ID"
    ]


def test_parser_output_must_be_nfc() -> None:
    decomposed = unicodedata.normalize("NFD", "café au lait")
    texts = {("doc.md", "v1"): decomposed}

    issues, _ = validate_questions([question("caf")], lambda d, v: texts[(d, v)])

    assert [i.code for i in issues] == ["TEXT_NOT_NFC"]


def test_load_questions_reads_directory_and_rejects_non_list(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("- id: a\n- id: b\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("- id: c\n", encoding="utf-8")
    assert [q["id"] for q in load_questions(tmp_path)] == ["a", "b", "c"]

    (tmp_path / "bad.yaml").write_text("id: a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="list of questions"):
        load_questions(tmp_path)


def test_load_provider_rejects_bad_spec() -> None:
    with pytest.raises(ValueError):
        load_provider("no_colon")
    assert callable(load_provider("os.path:join"))


def test_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    golden = tmp_path / "golden.yaml"
    golden.write_text(
        "- id: q1\n  category: single_section\n  evidence:\n"
        "    - {document: doc.md, version: v1, quote: 'Services expose Pods.'}\n",
        encoding="utf-8",
    )
    source = f"{__name__}:provider"

    assert main([str(golden), "--text-source", source, "--show-spans"]) == 0
    assert "q1 evidence[0]" in capsys.readouterr().out

    golden.write_text("- id: q1\n  category: single_section\n", encoding="utf-8")
    assert main([str(golden), "--text-source", source]) == 1
    assert main([str(tmp_path / "missing.yaml"), "--text-source", source]) == 2
