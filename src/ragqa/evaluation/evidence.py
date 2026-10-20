"""Validate the evidence quotes of the golden set against parser output.

Each question in a golden YAML file lists the evidence needed to answer it. A quote is
resolved in the parser output text of its document and version, and must match exactly
once. Whitespace differences are ignored. Offsets are derived here (characters of the
parser output text, end exclusive) and never written by hand.

File format: a YAML list of questions. Fields used here:

    - id: q001
      category: single_section        # "unanswerable" must have no evidence
      evidence:
        - document: concepts/workloads/pods/_index.md
          version: release-1.33
          quote: "A Pod is the smallest deployable unit"
          section_path: ["What is a Pod?"]   # optional, not checked yet
          required: true                     # optional, default true

Contract with the parser: its output is NFC-normalized Unicode, otherwise offsets and
quotes can disagree.

Usage:
    python -m ragqa.evaluation.evidence eval/golden --text-source my_module:get_text

The text source is a function (document, version) -> parser output text.
"""

import argparse
import importlib
import sys
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

TextProvider = Callable[[str, str], str]

UNANSWERABLE = "unanswerable"
REQUIRED_FIELDS = ("document", "version", "quote")


@dataclass(frozen=True)
class Span:
    start: int  # inclusive
    end: int  # exclusive


@dataclass(frozen=True)
class Issue:
    question_id: str
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.question_id}: {self.code}: {self.message}"


@dataclass(frozen=True)
class Resolution:
    spans: list[Span]  # every match, in offsets of the original text
    matched_prefix: int  # longest quote prefix that occurs in the text (when spans is empty)


def collapse_whitespace(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space.

    Returns the new text and, for each of its characters, the offset of the character it
    came from in the input, so matches can be mapped back to the original text.
    """
    chars: list[str] = []
    origin: list[int] = []
    in_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_space:
                chars.append(" ")
                origin.append(i)
            in_space = True
        else:
            chars.append(ch)
            origin.append(i)
            in_space = False
    return "".join(chars), origin


def prepare_quote(quote: str) -> str:
    return collapse_whitespace(unicodedata.normalize("NFC", quote))[0].strip()


def find_all(haystack: str, needle: str) -> list[int]:
    hits = []
    pos = haystack.find(needle)
    while pos != -1:
        hits.append(pos)
        pos = haystack.find(needle, pos + 1)
    return hits


def longest_prefix_found(haystack: str, needle: str) -> int:
    # A prefix occurs in the text only if all shorter prefixes do, so binary search works.
    lo, hi = 0, len(needle)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if needle[:mid] in haystack:
            lo = mid
        else:
            hi = mid - 1
    return lo


def resolve_quote(text: str, quote: str) -> Resolution:
    flat, origin = collapse_whitespace(text)
    needle = prepare_quote(quote)
    hits = find_all(flat, needle)
    spans = [Span(origin[h], origin[h + len(needle) - 1] + 1) for h in hits]
    prefix = 0 if spans else longest_prefix_found(flat, needle)
    return Resolution(spans=spans, matched_prefix=prefix)


def not_found_message(text: str, quote: str, matched_prefix: int) -> str:
    flat, _ = collapse_whitespace(text)
    needle = prepare_quote(quote)
    if matched_prefix == 0:
        return "quote not found, not even its first character"
    at = flat.find(needle[:matched_prefix])
    expected = needle[matched_prefix : matched_prefix + 30]
    actual = flat[at + matched_prefix : at + matched_prefix + 30]
    return (
        f"quote not found: the first {matched_prefix} of {len(needle)} characters match, "
        f"then the quote has {expected!r} but the text has {actual!r}"
    )


def check_evidence_item(
    question_id: str,
    index: int,
    item: Any,
    provider: TextProvider,
    cache: dict[tuple[str, str], str],
) -> tuple[list[Issue], Span | None]:
    label = f"evidence[{index}]"
    if not isinstance(item, dict):
        return [Issue(question_id, "BAD_EVIDENCE", f"{label} must be a mapping")], None
    missing = [f for f in REQUIRED_FIELDS if not item.get(f)]
    if missing:
        return [Issue(question_id, "MISSING_FIELD", f"{label} lacks {', '.join(missing)}")], None

    key = (str(item["document"]), str(item["version"]))
    if key not in cache:
        try:
            cache[key] = provider(*key)
        except (OSError, KeyError, ValueError) as exc:
            msg = f"{label}: no text for {key[0]} at {key[1]}: {exc}"
            return [Issue(question_id, "TEXT_UNAVAILABLE", msg)], None
    text = cache[key]

    if not unicodedata.is_normalized("NFC", text):
        msg = f"{label}: parser output for {key[0]} at {key[1]} is not NFC-normalized"
        return [Issue(question_id, "TEXT_NOT_NFC", msg)], None
    if not prepare_quote(str(item["quote"])):
        return [Issue(question_id, "EMPTY_QUOTE", f"{label} quote is empty")], None

    resolution = resolve_quote(text, str(item["quote"]))
    if not resolution.spans:
        msg = f"{label}: " + not_found_message(text, str(item["quote"]), resolution.matched_prefix)
        return [Issue(question_id, "NOT_FOUND", msg)], None
    if len(resolution.spans) > 1:
        spans = ", ".join(f"[{s.start}, {s.end})" for s in resolution.spans)
        msg = f"{label}: quote matches {len(resolution.spans)} times at {spans}; lengthen it"
        return [Issue(question_id, "AMBIGUOUS", msg)], None
    return [], resolution.spans[0]


def validate_questions(
    questions: list[Any], provider: TextProvider
) -> tuple[list[Issue], dict[tuple[str, int], Span]]:
    """Check every question. Returns the issues and the resolved span of each good quote."""
    issues: list[Issue] = []
    spans: dict[tuple[str, int], Span] = {}
    seen_ids: set[str] = set()
    cache: dict[tuple[str, str], str] = {}

    for position, question in enumerate(questions):
        if not isinstance(question, dict) or not question.get("id"):
            issues.append(Issue(f"#{position}", "BAD_QUESTION", "needs a mapping with an id"))
            continue
        qid = str(question["id"])
        if qid in seen_ids:
            issues.append(Issue(qid, "DUPLICATE_ID", "id is used more than once"))
        seen_ids.add(qid)

        evidence = question.get("evidence") or []
        if question.get("category") == UNANSWERABLE:
            if evidence:
                issues.append(
                    Issue(qid, "UNEXPECTED_EVIDENCE", "unanswerable question has evidence")
                )
            continue
        if not evidence:
            issues.append(Issue(qid, "NO_EVIDENCE", "answerable question has no evidence"))
            continue
        if not isinstance(evidence, list):
            issues.append(Issue(qid, "BAD_EVIDENCE", "evidence must be a list"))
            continue
        for i, item in enumerate(evidence):
            item_issues, span = check_evidence_item(qid, i, item, provider, cache)
            issues.extend(item_issues)
            if span is not None:
                spans[(qid, i)] = span
    return issues, spans


def load_questions(path: Path) -> list[Any]:
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    questions: list[Any] = []
    for file in files:
        data = yaml.safe_load(file.read_text(encoding="utf-8")) or []
        if not isinstance(data, list):
            raise ValueError(f"{file}: the top level must be a list of questions")
        questions.extend(data)
    return questions


def load_provider(spec: str) -> TextProvider:
    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError("text source must look like 'package.module:function'")
    provider = getattr(importlib.import_module(module_name), attr)
    if not callable(provider):
        raise ValueError(f"{spec} is not callable")
    return provider


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate golden-set evidence quotes.")
    parser.add_argument("path", type=Path, help="a golden YAML file or a directory of them")
    parser.add_argument(
        "--text-source",
        required=True,
        help="'package.module:function' returning parser output text for (document, version)",
    )
    parser.add_argument("--show-spans", action="store_true", help="print resolved offsets")
    args = parser.parse_args(argv)

    try:
        questions = load_questions(args.path)
        provider = load_provider(args.text_source)
    except (OSError, ValueError, ImportError, AttributeError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    issues, spans = validate_questions(questions, provider)
    for issue in issues:
        print(issue)
    if args.show_spans:
        for (qid, index), span in sorted(spans.items()):
            print(f"{qid} evidence[{index}]: [{span.start}, {span.end})")
    print(f"{len(questions)} questions, {len(spans)} quotes resolved, {len(issues)} issues")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
