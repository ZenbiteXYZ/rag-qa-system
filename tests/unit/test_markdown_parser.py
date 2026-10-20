import pytest

from ragqa.domain.documents import ParsedDocument
from ragqa.ingestion.errors import PermanentIngestionError
from ragqa.ingestion.parsers import MarkdownParser
from ragqa.ingestion.parsers.markdown_parser import split_front_matter, strip_shortcodes

# Shaped like a Kubernetes docs page.
SOURCE = """---
title: Pods
content_type: concept
weight: 10
reviewers:
- erictune
tags: [workloads, pods]
---

<!-- overview -->

_Pods_ are the smallest deployable units of computing that you can create and manage in
Kubernetes. A {{< glossary_tooltip text="Pod" term_id="pod" >}} runs on a node.

{{< note >}}
Restarting a container in a Pod should not be confused with restarting a Pod.
{{< /note >}}

## What is a Pod?

See the [Pod lifecycle](/docs/concepts/workloads/pods/pod-lifecycle/) and `kube_proxy` flags.

- a list item with **bold**
  - nested item
- second item

1. first step
2. second step

| Field | Description |
|-------|-------------|
| `spec.containers` | List of containers |

### Example

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: {{< param "name" >}}
```

![A diagram of a Pod](/images/pod.svg)
"""


@pytest.fixture
def doc() -> ParsedDocument:
    return MarkdownParser().parse(SOURCE.encode(), filename="_index.md")


def test_metadata_and_title_from_front_matter(doc: ParsedDocument) -> None:
    assert doc.title == "Pods"
    assert doc.source_format == "md"
    assert doc.metadata.doc_type == "concept"
    assert doc.metadata.tags == ("workloads", "pods")
    assert doc.front_matter["weight"] == 10
    assert "content_type" not in doc.text


def test_inline_markup_and_shortcodes_render_as_a_reader_sees_them(
    doc: ParsedDocument,
) -> None:
    assert "Pods are the smallest deployable units of computing" in doc.text
    assert "A Pod runs on a node." in doc.text
    assert "Restarting a container in a Pod should not be confused" in doc.text
    assert "See the Pod lifecycle and kube_proxy flags." in doc.text
    assert "{{" not in doc.text.replace("{{< param", "")  # only the one inside code stays
    assert "overview" not in doc.text  # HTML comment
    assert "A diagram of a Pod" in doc.text  # image alt text


def test_lists_and_tables(doc: ParsedDocument) -> None:
    assert "- a list item with bold\n  - nested item\n- second item" in doc.text
    assert "1. first step\n2. second step" in doc.text
    assert "Field | Description\nspec.containers | List of containers" in doc.text


def test_sections_follow_the_headings(doc: ParsedDocument) -> None:
    paths = [s.heading_path for s in doc.sections]

    assert paths == [(), ("What is a Pod?",), ("What is a Pod?", "Example")]
    assert doc.text[doc.sections[1].start :].startswith("What is a Pod?")
    assert doc.section_path_at(doc.text.index("kind: Pod")) == ("What is a Pod?", "Example")


def test_code_blocks_are_spans_and_keep_their_content(doc: ParsedDocument) -> None:
    (code,) = doc.code_spans
    block = doc.text[code.start : code.end]

    assert code.language == "yaml"
    assert block.startswith("apiVersion: v1")
    assert block.endswith('name: {{< param "name" >}}')  # shortcodes inside code are kept
    assert "```" not in doc.text


def test_title_falls_back_to_first_h1_then_filename() -> None:
    parser = MarkdownParser()

    assert parser.parse(b"# Services\n\ntext", filename="x.md").title == "Services"
    assert parser.parse(b"just text", filename="network-policies.md").title == "network policies"


def test_document_without_headings_is_one_section() -> None:
    doc = MarkdownParser().parse(b"Line one\n\nLine two", filename="a.md")

    assert len(doc.sections) == 1
    assert doc.text[doc.sections[0].start : doc.sections[0].end] == "Line one\n\nLine two"


def test_crlf_and_bom_are_normalized() -> None:
    doc = MarkdownParser().parse("﻿# T\r\n\r\nbody\r\n".encode(), filename="a.md")

    assert doc.text == "T\n\nbody"


@pytest.mark.parametrize(
    "data,message",
    [
        (b"\xff\xfe\x00bad", "UTF-8"),
        (b"", "no text"),
        (b"---\ntitle: x\n---\n<!-- c -->", "no text"),
    ],
)
def test_unusable_input_is_a_permanent_error(data: bytes, message: str) -> None:
    with pytest.raises(PermanentIngestionError, match=message):
        MarkdownParser().parse(data, filename="a.md")


def test_horizontal_rule_at_top_is_not_front_matter() -> None:
    front, body = split_front_matter("---\nSome prose, not YAML: really\n- but a list\n---\nrest")

    assert front == {}
    assert body.startswith("---")


def test_strip_shortcodes_leaves_tilde_fences_alone() -> None:
    source = "a {{< note >}}b{{< /note >}}\n~~~\n{{< keep >}}\n~~~\nc {{% x %}}"

    assert strip_shortcodes(source) == "a b\n~~~\n{{< keep >}}\n~~~\nc "


def test_glossary_tooltip_without_text_falls_back_to_term_id() -> None:
    source = b'If you use a {{< glossary_tooltip term_id="deployment" >}} to run your app.'

    doc = MarkdownParser().parse(source, filename="a.md")

    assert doc.text == "If you use a deployment to run your app."


def test_hugo_heading_ids_are_removed() -> None:
    doc = MarkdownParser().parse(
        b"## Port definitions {#field-spec-ports}\n\ntext", filename="a.md"
    )

    assert doc.sections[0].heading_path == ("Port definitions",)
    assert "{#" not in doc.text
