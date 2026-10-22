"""The synthetic `author` metadata of the corpus (ADR-0007, ML_DESIGN section 3).

The Kubernetes pages have no reliable author, so each page path gets one from a fixed list with
a fixed seed. The same path has the same author in every release. The generated mapping is
committed (`eval/corpus/authors.json`), so the ground truth of author-filter questions is
reproducible. These questions test the filter mechanism, not natural cues in the text.
"""

import json
import random
from collections.abc import Iterable
from pathlib import Path

DEFAULT_SEED = 20261014  # the date of ADR-0007 and ADR-0019
AUTHORS = (
    "A. Ivanova",
    "I. Petrov",
    "M. Sokolova",
    "D. Kuznetsov",
    "E. Smirnova",
    "N. Volkov",
    "O. Morozova",
    "S. Lebedev",
)


def make_author_mapping(
    paths: Iterable[str], *, seed: int = DEFAULT_SEED, authors: tuple[str, ...] = AUTHORS
) -> dict[str, str]:
    """One author per distinct path. Sorted first, so the result does not depend on the order
    the paths come in."""
    rng = random.Random(seed)
    return {path: rng.choice(authors) for path in sorted(set(paths))}


def paths_from_manifests(directory: Path) -> set[str]:
    """Page paths of every prepared release (`eval/corpus/release-*.json`)."""
    paths: set[str] = set()
    for manifest in sorted(directory.glob("release-*.json")):
        paths |= set(json.loads(manifest.read_text(encoding="utf-8"))["files"])
    return paths


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Write the synthetic author mapping.")
    parser.add_argument("--manifests", type=Path, default=Path("eval/corpus"))
    parser.add_argument("--out", type=Path, default=Path("eval/corpus/authors.json"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"{args.out} exists; delete it on purpose to regenerate")
        return 1
    mapping = make_author_mapping(paths_from_manifests(args.manifests), seed=args.seed)
    document = {"seed": args.seed, "authors": list(AUTHORS), "mapping": mapping}
    args.out.write_text(json.dumps(document, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(mapping)} pages written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
