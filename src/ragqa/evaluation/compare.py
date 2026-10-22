"""Paired comparison of two runs (ML_DESIGN section 4, 'Repetition and uncertainty').

The same questions are compared question by question. Reported next to the means: how many
questions got better, worse or stayed the same, a bootstrap interval over questions, and the
run-to-run noise measured from the repetitions. A difference is only called distinguishable
when its interval excludes zero and it is larger than the noise.
"""

import random
from collections.abc import Mapping
from dataclasses import dataclass

_EPS = 1e-9


@dataclass(frozen=True)
class Paired:
    metric: str
    n: int  # questions with a value in both runs
    excluded: int  # questions with a value in only one of them
    mean_base: float
    mean_other: float
    mean_diff: float
    ci_low: float
    ci_high: float
    better: int
    worse: int
    unchanged: int
    noise: float | None  # mean absolute difference between repetitions, larger of the two runs
    distinguishable: bool


def bootstrap_mean_ci(
    diffs: list[float], *, seed: int, resamples: int = 10_000, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of the mean over questions, resampling questions with replacement."""
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    low = means[int(resamples * (1 - level) / 2)]
    high = means[min(resamples - 1, int(resamples * (1 + level) / 2))]
    return low, high


def paired_comparison(
    base: Mapping[str, Mapping[str, float | None]],
    other: Mapping[str, Mapping[str, float | None]],
    metric: str,
    *,
    noise: float | None = None,
    seed: int = 0,
    resamples: int = 10_000,
) -> Paired | None:
    """None when no question has the metric in both runs."""
    ids = sorted(set(base) & set(other))
    both = [
        (base[i][metric], other[i][metric])
        for i in ids
        if base[i].get(metric) is not None and other[i].get(metric) is not None
    ]
    pairs = [(a, b) for a, b in both if a is not None and b is not None]
    if not pairs:
        return None
    one_sided = sum(
        1
        for i in set(base) | set(other)
        if (base.get(i, {}).get(metric) is None) != (other.get(i, {}).get(metric) is None)
    )
    diffs = [b - a for a, b in pairs]
    mean_diff = sum(diffs) / len(diffs)
    low, high = bootstrap_mean_ci(diffs, seed=seed, resamples=resamples)
    excludes_zero = low > 0 or high < 0
    return Paired(
        metric=metric,
        n=len(pairs),
        excluded=one_sided,
        mean_base=sum(a for a, _ in pairs) / len(pairs),
        mean_other=sum(b for _, b in pairs) / len(pairs),
        mean_diff=mean_diff,
        ci_low=low,
        ci_high=high,
        better=sum(d > _EPS for d in diffs),
        worse=sum(d < -_EPS for d in diffs),
        unchanged=sum(abs(d) <= _EPS for d in diffs),
        noise=noise,
        distinguishable=excludes_zero and (noise is None or abs(mean_diff) > noise),
    )
