# Golden set

Hand-written questions (ADR-0005), YAML lists in this folder. 42 in total: 6 in each of 7 categories, 3 `dev` and 3 `test` per category (ADR-0019).

Each question (`ragqa.evaluation.golden.GoldenQuestion`):

```yaml
- id: sec-01                      # unique
  category: single_section        # single_section, multi_section, exact_token, metadata_cue,
                                  # version_sensitive, follow_up, unanswerable
  split: dev                      # dev or test
  question: How do I drain a node?
  history:                        # follow_up only: earlier turns, oldest first
    - {role: user, content: ...}
    - {role: assistant, content: ...}
  expected_filters:               # what Self-Query should extract (all optional)
    author: I. Petrov             # doc_type, tags, author, language, date_from, date_to, document
  release: release-1.32           # version_sensitive: the release the question asks about
  evidence:                       # none for unanswerable questions
    - document: tasks/administer-cluster/safely-drain-node.md   # path below the release folder
      version: release-1.34                                     # release folder name
      quote: "verbatim, one to three sentences, matches exactly once"
      required: true              # false marks an alternative that is not counted
  reference_answer: ...           # for an unanswerable question, say the documents do not contain it
```

Check a file against the corpus (every quote must match exactly once, and the set must have the right shape):

    python -m ragqa.evaluation validate-golden eval/golden --corpus ~/data/corpora/kubernetes/corpus

The judge calibration items (30 hand-labeled answers, `calibration.yaml`) live here too; see `ragqa.evaluation.calibrate`.
