"""Prepare the Kubernetes docs corpus: Hugo shortcodes become plain Markdown (ADR-0006).

The site builds its pages from shortcodes that a Markdown parser cannot see through:
code samples and includes come from other files, section headings from translation
strings, and feature states from data files. This step renders them the way the site's
templates do, so the corpus holds the text a reader sees. It is run once per release:

    python -m ragqa.evaluation.corpus --raw ~/data/corpora/kubernetes/raw \\
        --out ~/data/corpora/kubernetes/corpus --versions release-1.34

`--raw/<version>` is a sparse checkout of kubernetes/website with `content/en/docs`
(concepts, tasks, reference/command-line-tools-reference/feature-gates), `content/en/examples`,
`content/en/includes`, `i18n/en` and `hugo.toml`. Output goes to `--out/<version>/` with the
paths below `content/en/docs`, which is the layout `ragqa.evaluation.text_sources` reads.
Shortcodes without a rule here are dropped and counted in the manifest. Anything that
should resolve but does not (a missing example file) is an error.
"""

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tomllib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.ingestion.parsers import MarkdownParser
from ragqa.ingestion.parsers.markdown_parser import split_front_matter

SOURCE_URL = "https://github.com/kubernetes/website"
LICENSE = "CC-BY-4.0"
_SHORTCODE = re.compile(r"\{\{[<%]\s*(/?)\s*([\w.-]+)(.*?)\s*[>%]\}\}", re.DOTALL)
_LANGUAGES = {"yml": "yaml", "py": "python", "sh": "shell"}
_ALERTS = {"note", "caution", "warning"}
_TRANSPARENT = {"tabs", "table", "example", "code"}  # wrappers whose inner text is kept as is
_COMMENT = re.compile(r"\{\{[<%]\s*comment\s*[>%]\}\}.*?\{\{[<%]\s*/comment\s*[>%]\}\}", re.DOTALL)
_MAX_INCLUDE_DEPTH = 5
_FEATURE_GATES = "reference/command-line-tools-reference/feature-gates"


class CorpusError(Exception):
    """Something that should resolve did not. The manifest lists every problem."""


@dataclass
class Site:
    version: str  # `params.version` in hugo.toml, the release of the branch
    latest: str  # `params.latest`
    strings: Mapping[str, str]  # i18n/en/en.toml, key -> text
    patches: Mapping[str, str]  # minor release -> latest patch, from data/releases/schedule.yaml

    def t(self, key: str) -> str:
        return self.strings[key]


@dataclass
class Report:
    dropped: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)


def load_site(raw: Path) -> Site:
    hugo = tomllib.loads((raw / "hugo.toml").read_text(encoding="utf-8"))
    i18n = tomllib.loads((raw / "i18n/en/en.toml").read_text(encoding="utf-8"))
    strings = {k: v["other"] for k, v in i18n.items() if isinstance(v, dict) and "other" in v}
    schedule = yaml.safe_load((raw / "data/releases/schedule.yaml").read_text(encoding="utf-8"))
    patches = {
        str(item["release"]): str(item["previousPatches"][0]["release"])
        for item in schedule["schedules"]
        if item.get("previousPatches")
    }
    return Site(
        version=hugo["params"]["version"],
        latest=hugo["params"]["latest"],
        strings=strings,
        patches=patches,
    )


def parse_args(raw_args: str) -> tuple[list[str], dict[str, str]]:
    """Split shortcode arguments into positional values and `name=value` pairs."""
    positional: list[str] = []
    named: dict[str, str] = {}
    for token in shlex.split(raw_args):
        name, sep, value = token.partition("=")
        if sep and re.fullmatch(r"\w+", name):
            named[name] = value
        else:
            positional.append(token)
    return positional, named


def _minor(version: str) -> tuple[str, int]:
    major, minor = version.lstrip("v").split(".")[:2]
    return major, int(minor)


def _skew(site: Site, args: list[str]) -> str:
    kind = args[0] if args else ""
    sep = args[2] if len(args) > 2 else "."
    major, latest_minor = _minor(site.latest)
    _, current_minor = _minor(site.version)
    add = int(args[1]) if len(args) > 1 and args[1].lstrip("-").isdigit() else 0
    values = {
        "latestVersion": site.latest.lstrip("v"),
        "nextMinorVersion": f"{major}.{latest_minor + 1}",
        "prevMinorVersion": f"{major}.{latest_minor - 1}",
        "oldestMinorVersion": f"{major}.{latest_minor - 2}",
        "latestVersionAddMinor": f"{major}{sep}{latest_minor + add}",
        "currentVersion": site.version.lstrip("v"),
        "currentPatchVersion": site.patches.get(
            site.version.lstrip("v"), f"{site.version.lstrip('v')}.0"
        ),
        "currentVersionAddMinor": f"{major}{sep}{current_minor + add}",
    }
    if kind not in values:
        raise KeyError(f"skew {kind}")
    return values[kind]


class Renderer:
    def __init__(self, raw: Path, site: Site, report: Report) -> None:
        self.docs = raw / "content/en/docs"
        self.examples = raw / "content/en/examples"
        self.includes = raw / "content/en/includes"
        self.site = site
        self.report = report

    def render(self, body: str, *, page: Path, front: Mapping[str, Any], depth: int = 0) -> str:
        body = _COMMENT.sub("", body)  # hidden on the site
        # Hugo expands shortcodes everywhere, including inside fenced code blocks, which is
        # how `{{< skew currentPatchVersion >}}` ends up in install commands.
        return _SHORTCODE.sub(lambda m: self._shortcode(m, page, front, depth), body)

    def _error(self, message: str, page: Path) -> str:
        self.report.errors.append(f"{page.relative_to(self.docs)}: {message}")
        return ""

    def _shortcode(
        self, match: re.Match[str], page: Path, front: Mapping[str, Any], depth: int
    ) -> str:
        closing, name, raw_args = match.groups()
        try:
            args, named = parse_args(raw_args)
        except ValueError:
            return self._error(f"cannot read the arguments of {match.group(0)[:60]!r}", page)

        if name == "glossary_tooltip":
            return match.group(0)  # the parser renders these
        if name in _TRANSPARENT:
            return ""
        if name in _ALERTS:
            return "\n" if closing else f"\n\n**{self.site.t(name)}** "
        if name == "alert":
            title = named.get("title", "").strip()
            return "\n" if closing else (f"\n\n**{title}** " if title else "\n\n")
        if name == "details":
            summary = named.get("summary", "").strip()
            return "\n\n" if closing or not summary else f"\n\n**{summary}**\n\n"
        if name == "tab":
            label = named.get("name", "").strip()
            return "\n\n" if closing or not label else f"\n\n**{label}**\n\n"
        if name in {"highlight", "mermaid"}:
            language = "mermaid" if name == "mermaid" else (args[0] if args else "")
            return "\n```\n" if closing else f"\n```{language}\n"
        if closing:
            self.report.dropped[f"/{name}"] += 1
            return ""
        handler = {
            "code_sample": self._code_sample,
            "include": self._include,
            "heading": self._heading,
            "feature-state": self._feature_state,
            "skew": self._skew,
            "param": self._param,
            "version-check": self._version_check,
            "figure": self._figure,
            "glossary_definition": self._glossary_definition,
        }.get(name)
        if handler is None:
            self.report.dropped[name] += 1
            return ""
        try:
            return handler(args, named, page, front, depth)
        except (KeyError, OSError, ValueError) as exc:
            return self._error(f"{name}: {exc}", page)

    def _code_sample(
        self,
        args: list[str],
        named: dict[str, str],
        page: Path,
        front: Mapping[str, Any],
        depth: int,
    ) -> str:
        file = named["file"]
        path = self.examples / file.lstrip("/")
        language = named.get("language") or _LANGUAGES.get(path.suffix[1:], path.suffix[1:])
        return f"\n```{language}\n{path.read_text(encoding='utf-8').rstrip()}\n```\n"

    def _include(
        self,
        args: list[str],
        named: dict[str, str],
        page: Path,
        front: Mapping[str, Any],
        depth: int,
    ) -> str:
        if depth >= _MAX_INCLUDE_DEPTH:
            raise ValueError("includes nested too deeply")
        name = args[0]
        candidates = [
            self.includes / name,
            *sorted(self.includes.glob(f"{name}*")),
            page.parent / name,
        ]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"no include named {name!r}")
        _, body = split_front_matter(path.read_text(encoding="utf-8"))
        return "\n\n" + self.render(body, page=page, front=front, depth=depth + 1) + "\n\n"

    def _heading(
        self,
        args: list[str],
        named: dict[str, str],
        page: Path,
        front: Mapping[str, Any],
        depth: int,
    ) -> str:
        return self.site.t(f"{args[0]}_heading")

    def _feature_state(
        self,
        args: list[str],
        named: dict[str, str],
        page: Path,
        front: Mapping[str, Any],
        depth: int,
    ) -> str:
        label = f"{self.site.t('feature_state')} {self.site.t('feature_state_kubernetes_label')}"
        gate = named.get("feature_gate_name")
        if gate is None:
            state = named["state"]
            if state not in {"alpha", "beta", "deprecated", "stable"}:
                raise ValueError(f"invalid state {state!r}")
            version = named.get("for_k8s_version") or self.site.version
            return f"\n\n{label} {version} [{state}]\n\n"
        gate_file = self.docs / _FEATURE_GATES / f"{gate}.md"
        data, _ = split_front_matter(gate_file.read_text(encoding="utf-8"))
        stage = data["stages"][-1]
        default = self._gate_default(bool(stage.get("defaultValue")))
        return f"\n\n{label} v{stage['fromVersion']} [{stage['stage']}] {default}\n\n"

    def _gate_default(self, enabled: bool) -> str:
        """Older releases use one template string with the value, newer ones two strings."""
        template = self.site.t("feature_gate_enabled")
        if "{{ .enabled }}" in template:
            return template.replace("{{ .enabled }}", "true" if enabled else "false")
        return self.site.t("feature_gate_enabled" if enabled else "feature_gate_disabled")

    def _skew(
        self,
        args: list[str],
        named: dict[str, str],
        page: Path,
        front: Mapping[str, Any],
        depth: int,
    ) -> str:
        return _skew(self.site, args)

    def _param(
        self,
        args: list[str],
        named: dict[str, str],
        page: Path,
        front: Mapping[str, Any],
        depth: int,
    ) -> str:
        key = args[0]
        if key == "version":
            return self.site.version
        if key in front:
            return str(front[key])
        raise KeyError(f"param {key}")

    def _version_check(
        self,
        args: list[str],
        named: dict[str, str],
        page: Path,
        front: Mapping[str, Any],
        depth: int,
    ) -> str:
        minimum = front.get("min-kubernetes-server-version")
        text = ""
        if minimum:
            key = (
                "version_check_mustbe"
                if str(minimum) == self.site.version
                else "version_check_mustbeorlater"
            )
            text = f"{self.site.t(key)}{minimum}. "
        return f"\n\n{text}{self.site.t('version_check_tocheck')}`kubectl version`.\n\n"

    def _glossary_definition(
        self,
        args: list[str],
        named: dict[str, str],
        page: Path,
        front: Mapping[str, Any],
        depth: int,
    ) -> str:
        length = named["length"]
        if length not in {"short", "long", "all"}:
            raise ValueError(f"invalid length {length!r}")
        path = self.docs / "reference/glossary" / f"{named['term_id']}.md"
        _, body = split_front_matter(path.read_text(encoding="utf-8"))
        paragraphs = [p.strip() for p in body.strip().split("\n\n") if p.strip()]
        if not paragraphs:
            raise ValueError(f"empty glossary entry {named['term_id']!r}")
        prepend = named.get("prepend")
        if prepend:
            paragraphs[0] = f"{prepend} {paragraphs[0][0].lower()}{paragraphs[0][1:]}"
        text = paragraphs[0] if length == "short" else "\n\n".join(paragraphs)
        return "\n\n" + self.render(text, page=page, front=front, depth=depth + 1) + "\n\n"

    def _figure(
        self,
        args: list[str],
        named: dict[str, str],
        page: Path,
        front: Mapping[str, Any],
        depth: int,
    ) -> str:
        text = named.get("caption") or named.get("title") or named.get("alt") or ""
        return f"\n\n{text}\n\n" if text else ""


def prepare_version(
    raw_root: Path, out_root: Path, version: str, sections: tuple[str, ...] = ("concepts", "tasks")
) -> dict[str, Any]:
    raw = raw_root / version
    out = out_root / version
    if out.exists():
        raise CorpusError(f"{out} already exists; remove it first (it may be read-only)")
    site = load_site(raw)
    report = Report()
    renderer = Renderer(raw, site, report)
    parser = MarkdownParser()

    files: dict[str, str] = {}
    skipped_empty: list[str] = []
    failed: dict[str, str] = {}
    sources = sorted(p for s in sections for p in (renderer.docs / s).rglob("*.md"))
    for source in sources:
        relative = source.relative_to(renderer.docs)
        front, body = split_front_matter(source.read_text(encoding="utf-8"))
        rendered = renderer.render(body, page=source, front=front)
        header = (
            "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n\n"
            if front
            else ""
        )
        text = header + rendered.strip() + "\n"
        try:
            parser.parse(text.encode(), filename=source.name)
        except PermanentIngestionError as exc:
            if "no text" in str(exc):
                skipped_empty.append(str(relative))  # section index pages have only front matter
            else:
                failed[str(relative)] = str(exc)
            continue
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        files[str(relative)] = hashlib.sha256(text.encode()).hexdigest()

    manifest: dict[str, Any] = {
        "source": SOURCE_URL,
        "branch": version,
        "commit": subprocess.run(
            ["git", "-C", str(raw), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "license": LICENSE,
        "site": {"version": site.version, "latest": site.latest},
        "sections": list(sections),
        "counts": {
            "source_pages": len(sources),
            "written": len(files),
            "skipped_empty": len(skipped_empty),
            "failed": len(failed),
        },
        "dropped_shortcodes": dict(sorted(report.dropped.items())),
        "skipped_empty": skipped_empty,
        "failed": failed,
        "errors": report.errors,
        "files": files,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if report.errors or failed:
        problems = f"{len(report.errors)} unresolved shortcodes, {len(failed)} unparseable pages"
        raise CorpusError(f"{version}: {problems}; see MANIFEST.json")
    return manifest


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description="Prepare the Kubernetes docs corpus.")
    cli.add_argument("--raw", type=Path, required=True, help="folder with one checkout per version")
    cli.add_argument("--out", type=Path, required=True, help="folder for the prepared corpus")
    cli.add_argument("--versions", nargs="+", required=True, help="for example release-1.34")
    args = cli.parse_args(argv)
    status = 0
    for version in args.versions:
        try:
            counts = prepare_version(args.raw, args.out, version)["counts"]
            print(f"{version}: {counts}")
        except CorpusError as exc:
            print(f"error: {exc}", file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
