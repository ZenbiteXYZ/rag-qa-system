import json
import re
import subprocess
from pathlib import Path

import pytest

from ragqa.evaluation.corpus import CorpusError, parse_args, prepare_version

GLOSSARY_POD = "---\ntitle: Pod\n---\nThe smallest unit.\n\nMore about pods.\n"
GATES = "reference/command-line-tools-reference/feature-gates"
HUGO = '[params]\nversion = "v1.34"\nlatest = "v1.36"\n'
I18N = """
[note]
other = "Note:"
[whatsnext_heading]
other = "What's next"
[feature_state]
other = "FEATURE STATE:"
[feature_state_kubernetes_label]
other = "Kubernetes"
[feature_gate_enabled]
other = "(enabled by default)"
[feature_gate_disabled]
other = "(disabled by default)"
[version_check_mustbeorlater]
other = "Your Kubernetes server must be at or later than version "
[version_check_tocheck]
other = "To check the version, enter "
"""
SCHEDULE = """
schedules:
- release: "1.34"
  previousPatches:
  - release: 1.34.2
  - release: 1.34.1
"""
GATE = """---
title: TopologyManager
stages:
  - stage: beta
    defaultValue: true
    fromVersion: "1.18"
  - stage: stable
    defaultValue: false
    fromVersion: "1.27"
---
Gate text.
"""
PAGE = """---
title: Probes
content_type: task
min-kubernetes-server-version: v1.30
---

## {{% heading "whatsnext" %}}

{{< feature-state for_k8s_version="v1.25" state="stable" >}}
{{< feature-state feature_gate_name="TopologyManager" >}}

{{< note >}}
Mind the **gap**.
{{< /note >}}

{{< tabs name="x" >}}
{{% tab name="Linux" %}}
Use bash.
{{% /tab %}}
{{< /tabs >}}

{{% code_sample file="/pods/probe.yaml" %}}

{{< include "prereqs.md" >}}

{{< comment >}}
Hidden text.
{{< /comment >}}

Client is v{{< skew currentPatchVersion >}} and the next is {{< skew nextMinorVersion >}}.

```shell
download v{{< skew currentVersion >}}
```

{{< glossary_definition term_id="pod" length="short" prepend="A Pod is" >}}

{{< version-check >}}

{{< thirdparty-content >}}
"""


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    root = tmp_path / "raw" / "release-1.34"
    files = {
        "hugo.toml": HUGO,
        "i18n/en/en.toml": I18N,
        "data/releases/schedule.yaml": SCHEDULE,
        "content/en/docs/concepts/probes.md": PAGE,
        f"content/en/docs/{GATES}/TopologyManager.md": GATE,
        "content/en/docs/concepts/_index.md": "---\ntitle: Concepts\n---\n",
        "content/en/docs/reference/glossary/pod.md": GLOSSARY_POD,
        "content/en/examples/pods/probe.yaml": "kind: Pod\n",
        "content/en/includes/prereqs.md": "You need a cluster. {{< note >}}Really.{{< /note >}}\n",
    }
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    for cmd in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"],
    ):
        subprocess.run(["git", "-C", str(root), *cmd], check=True)
    return tmp_path / "raw"


def test_parse_args() -> None:
    assert parse_args('"a b" file=x.yaml name="Linux x" 1 "-"') == (
        ["a b", "1", "-"],
        {"file": "x.yaml", "name": "Linux x"},
    )


def test_renders_shortcodes_the_way_the_site_does(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "corpus"

    manifest = prepare_version(raw, out, "release-1.34")
    text = (out / "release-1.34" / "concepts" / "probes.md").read_text(encoding="utf-8")

    assert "## What's next" in text
    assert "FEATURE STATE: Kubernetes v1.25 [stable]" in text
    assert "FEATURE STATE: Kubernetes v1.27 [stable] (disabled by default)" in text
    assert "**Note:** \nMind the **gap**." in text
    assert re.search(r"\*\*Linux\*\*\n+Use bash\.", text)
    assert "```yaml\nkind: Pod\n```" in text  # a leading slash in the file name is fine
    assert "You need a cluster. \n\n**Note:** Really." in text
    assert "Hidden text" not in text
    assert "Client is v1.34.2 and the next is 1.37." in text
    assert "download v1.34\n" in text  # shortcodes are expanded inside code blocks too
    assert "A Pod is the smallest unit." in text
    assert (
        "Your Kubernetes server must be at or later than version v1.30. To check the version"
        in text
    )
    assert "{{" not in text
    assert manifest["dropped_shortcodes"] == {"thirdparty-content": 1}


def test_manifest_and_skipped_pages(raw: Path, tmp_path: Path) -> None:
    out = tmp_path / "corpus"

    prepare_version(raw, out, "release-1.34")
    manifest = json.loads((out / "release-1.34" / "MANIFEST.json").read_text())

    assert manifest["counts"] == {"source_pages": 2, "written": 1, "skipped_empty": 1, "failed": 0}
    assert manifest["skipped_empty"] == ["concepts/_index.md"]
    assert len(manifest["commit"]) == 40
    assert manifest["license"] == "CC-BY-4.0"
    assert list(manifest["files"]) == ["concepts/probes.md"]


def test_missing_example_fails_loudly_but_writes_the_manifest(raw: Path, tmp_path: Path) -> None:
    (raw / "release-1.34/content/en/examples/pods/probe.yaml").unlink()
    out = tmp_path / "corpus"

    with pytest.raises(CorpusError, match="1 unresolved"):
        prepare_version(raw, out, "release-1.34")

    manifest = json.loads((out / "release-1.34" / "MANIFEST.json").read_text())
    assert "code_sample" in manifest["errors"][0]


def test_existing_output_is_never_overwritten(raw: Path, tmp_path: Path) -> None:
    (tmp_path / "corpus" / "release-1.34").mkdir(parents=True)

    with pytest.raises(CorpusError, match="already exists"):
        prepare_version(raw, tmp_path / "corpus", "release-1.34")


def test_older_release_with_a_single_feature_gate_string(raw: Path, tmp_path: Path) -> None:
    i18n = raw / "release-1.34/i18n/en/en.toml"
    single = '[feature_gate_enabled]\nother = "(enabled by default: {{ .enabled }})"'
    both = (
        '[feature_gate_enabled]\nother = "(enabled by default)"\n'
        '[feature_gate_disabled]\nother = "(disabled by default)"'
    )
    text = i18n.read_text().replace(both, single)
    i18n.write_text(text)
    out = tmp_path / "corpus"

    prepare_version(raw, out, "release-1.34")

    rendered = (out / "release-1.34/concepts/probes.md").read_text()
    assert "[stable] (enabled by default: false)" in rendered
