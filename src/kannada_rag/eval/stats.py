"""Statistical reporting for small evaluation sets.

With 25 questions, a difference of a few points between two models is well
inside sampling noise. Reporting bare means invites conclusions the data
cannot support, so every aggregate this project publishes carries an interval,
and every model-to-model comparison carries a paired test with a correction
for multiple comparisons.

Choices, and why:

*Bootstrap percentile intervals rather than normal-theory intervals.* Metric
distributions over 25 questions are bounded in [0, 1], skewed, and often have
a spike at zero (a response that shares no unigram with the reference). A
normal approximation puts mass outside the valid range; the bootstrap does not.

*Paired tests rather than unpaired.* Every model answers the same questions, so
the pairing is real and ignoring it throws away most of the power available at
this sample size.

*Wilcoxon signed-rank as the default.* It does not assume normal differences.
The paired t-test is provided alongside it, and both are reported so a reader
can see that the conclusion does not depend on the choice.

*Holm-Bonferroni correction.* Comparing five models pairwise is ten tests;
without correction the probability of at least one spurious "significant"
result is around 40%. Holm is uniformly more powerful than Bonferroni and
makes no independence assumption.

*Effect sizes.* A p-value says a difference is detectable, not that it matters.
Cliff's delta is reported because it is ordinal and robust at this sample size.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "Interval",
    "ComparisonResult",
    "bootstrap_ci",
    "describe",
    "paired_comparison",
    "holm_bonferroni",
    "cliffs_delta",
    "pairwise_model_comparison",
]


@dataclass
class Interval:
    mean: float
    low: float
    high: float
    std: float
    n: int
    level: float = 0.95

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.low:.3f}, {self.high:.3f}]"

    def to_dict(self) -> dict:
        return asdict(self)


def bootstrap_ci(
    values: Sequence[float],
    n_resamples: int = 10_000,
    level: float = 0.95,
    seed: int = 20240101,
    statistic=np.mean,
) -> Interval:
    """Percentile bootstrap confidence interval for a statistic.

    ``seed`` is fixed so that reported intervals are reproducible to the digit.
    """
    array = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if array.size == 0:
        return Interval(float("nan"), float("nan"), float("nan"), float("nan"), 0, level)
    if array.size == 1:
        value = float(array[0])
        return Interval(value, value, value, 0.0, 1, level)

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(n_resamples, array.size))
    replicates = statistic(array[indices], axis=1)

    alpha = (1 - level) / 2
    return Interval(
        mean=float(statistic(array)),
        low=float(np.quantile(replicates, alpha)),
        high=float(np.quantile(replicates, 1 - alpha)),
        std=float(np.std(array, ddof=1)),
        n=int(array.size),
        level=level,
    )


def describe(values: Sequence[float], **kwargs) -> dict:
    """Mean, standard deviation, n and bootstrap CI in one dictionary."""
    interval = bootstrap_ci(values, **kwargs)
    return {
        "mean": round(interval.mean, 4),
        "std": round(interval.std, 4),
        "ci_low": round(interval.low, 4),
        "ci_high": round(interval.high, 4),
        "n": interval.n,
    }


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> tuple[float, str]:
    """Cliff's delta and its conventional magnitude label.

    delta = P(a > b) - P(a < b), in [-1, 1]. Thresholds follow Romano et al.
    (2006): |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if x.size == 0 or y.size == 0:
        return float("nan"), "undefined"

    greater = int((x[:, None] > y[None, :]).sum())
    less = int((x[:, None] < y[None, :]).sum())
    delta = (greater - less) / (x.size * y.size)

    magnitude = abs(delta)
    if magnitude < 0.147:
        label = "negligible"
    elif magnitude < 0.33:
        label = "small"
    elif magnitude < 0.474:
        label = "medium"
    else:
        label = "large"
    return float(delta), label


@dataclass
class ComparisonResult:
    """One paired comparison between two conditions on the same questions."""

    name_a: str
    name_b: str
    metric: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_difference: float
    difference_ci: tuple[float, float]
    wilcoxon_p: float
    ttest_p: float
    cliffs_delta: float
    effect_label: str
    p_adjusted: float | None = None
    significant: bool | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["difference_ci"] = list(self.difference_ci)
        return payload

    def summary(self) -> str:
        stars = ""
        if self.p_adjusted is not None:
            stars = " *" if self.significant else " (n.s.)"
        return (
            f"{self.name_a} vs {self.name_b} on {self.metric}: "
            f"{self.mean_difference:+.3f} "
            f"[{self.difference_ci[0]:+.3f}, {self.difference_ci[1]:+.3f}], "
            f"p={self.p_adjusted if self.p_adjusted is not None else self.wilcoxon_p:.4f}"
            f", delta={self.cliffs_delta:+.2f} ({self.effect_label}){stars}"
        )


def paired_comparison(
    values_a: Sequence[float],
    values_b: Sequence[float],
    name_a: str = "A",
    name_b: str = "B",
    metric: str = "score",
    n_resamples: int = 10_000,
    seed: int = 20240101,
) -> ComparisonResult:
    """Paired comparison with both a nonparametric and a parametric test.

    The two sequences must be aligned question by question. Pairs where either
    value is missing are dropped, and the surviving count is reported.
    """
    from scipy import stats as sp

    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Unpaired inputs: {a.shape} vs {b.shape}")

    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    differences = a - b

    if a.size < 2:
        return ComparisonResult(
            name_a, name_b, metric, int(a.size),
            float(a.mean()) if a.size else float("nan"),
            float(b.mean()) if b.size else float("nan"),
            float(differences.mean()) if a.size else float("nan"),
            (float("nan"), float("nan")), float("nan"), float("nan"),
            float("nan"), "undefined",
        )

    if np.allclose(differences, 0):
        wilcoxon_p = 1.0
        ttest_p = 1.0
    else:
        wilcoxon_p = float(sp.wilcoxon(a, b, zero_method="wilcox").pvalue)
        ttest_p = float(sp.ttest_rel(a, b).pvalue)

    difference_ci = bootstrap_ci(differences, n_resamples=n_resamples, seed=seed)
    delta, label = cliffs_delta(a, b)

    return ComparisonResult(
        name_a=name_a,
        name_b=name_b,
        metric=metric,
        n_pairs=int(a.size),
        mean_a=float(a.mean()),
        mean_b=float(b.mean()),
        mean_difference=float(differences.mean()),
        difference_ci=(difference_ci.low, difference_ci.high),
        wilcoxon_p=wilcoxon_p,
        ttest_p=ttest_p,
        cliffs_delta=delta,
        effect_label=label,
    )


def holm_bonferroni(
    comparisons: list[ComparisonResult],
    alpha: float = 0.05,
    use: str = "wilcoxon",
) -> list[ComparisonResult]:
    """Apply the Holm step-down correction in place and return the list.

    Sorts p-values ascending and compares the i-th against ``alpha / (m - i)``,
    stopping at the first failure; every subsequent hypothesis is retained.
    """
    if not comparisons:
        return comparisons

    key = "wilcoxon_p" if use == "wilcoxon" else "ttest_p"
    order = sorted(
        range(len(comparisons)),
        key=lambda i: getattr(comparisons[i], key),
    )
    m = len(comparisons)

    running_max = 0.0
    rejected = True
    for position, index in enumerate(order):
        raw = getattr(comparisons[index], key)
        adjusted = min(1.0, (m - position) * raw)
        running_max = max(running_max, adjusted)  # enforce monotonicity
        comparisons[index].p_adjusted = running_max
        if rejected and running_max > alpha:
            rejected = False
        comparisons[index].significant = rejected and running_max <= alpha

    return comparisons


def pairwise_model_comparison(
    per_model_scores: dict[str, Sequence[float]],
    metric: str = "score",
    alpha: float = 0.05,
    seed: int = 20240101,
) -> list[ComparisonResult]:
    """All pairwise comparisons across models, Holm-corrected.

    This is the function that turns "model X scored higher" into a defensible
    claim. With five models there are ten comparisons, and at 25 questions most
    of them will not survive correction -- which is itself the finding worth
    reporting.
    """
    names = sorted(per_model_scores)
    comparisons: list[ComparisonResult] = []

    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            comparisons.append(
                paired_comparison(
                    per_model_scores[name_a],
                    per_model_scores[name_b],
                    name_a=name_a,
                    name_b=name_b,
                    metric=metric,
                    seed=seed,
                )
            )

    return holm_bonferroni(comparisons, alpha=alpha)
