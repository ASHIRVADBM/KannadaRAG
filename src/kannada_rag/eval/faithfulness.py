"""Faithfulness, attribution and abstention: does the answer match its source?

BERTScore cannot establish factual grounding. It measures semantic similarity
between a generated answer and a reference answer, and a fluent, confident,
entirely fabricated Kannada sentence about a Hoysala temple will score well
against a reference about the same temple. Any claim that retrieval "reduces
hallucination" therefore needs a measurement aimed at hallucination, not at
similarity.

Three complementary measurements are provided, in increasing order of cost and
of evidential weight:

``support_score`` (automatic)
    The fraction of an answer's content that is lexically traceable to the
    retrieved passages, measured by maximum character-n-gram overlap between
    each answer sentence and the supplied context. Cheap, runs on every
    question in every condition, and is directly interpretable: a grounded
    system should score high, an ungrounded one low.

``abstention`` (automatic)
    Behaviour on questions whose answers are *deliberately absent* from the
    corpus. A grounded system should decline; a system that answers anyway is
    hallucinating by construction, with no annotation required. This is the
    cleanest automatic evidence of grounding available, and it is why the
    benchmark includes an unanswerable tier.

``human_rating`` (manual)
    A per-claim judgement by annotators: is each claim supported by the cited
    passage, contradicted by it, or absent from it. The schema and agreement
    computation live here; the judgements themselves are collected with
    ``scripts/make_annotation_sheet.py``. Automatic proxies indicate, human
    judgements establish.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from ..text import character_ngrams, normalise_for_index, tokenise_for_metrics

__all__ = [
    "split_sentences",
    "support_score",
    "AbstentionResult",
    "score_abstention",
    "ClaimJudgement",
    "aggregate_human_judgements",
    "krippendorff_alpha_nominal",
]


_SENTENCE_END = re.compile(r"(?<=[।॥.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split Kannada text into sentences on danda, double danda or Latin stops."""
    text = normalise_for_index(text)
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]


def _ngram_containment(candidate: str, source: str, n: int = 4) -> float:
    """Fraction of the candidate's character n-grams that occur in the source.

    Containment rather than F-score: a one-sentence answer drawn faithfully
    from a long passage should score 1.0, and an F-score would penalise it for
    the passage's length.
    """
    candidate_grams = Counter(character_ngrams(candidate, n))
    if not candidate_grams:
        return 0.0
    source_grams = Counter(character_ngrams(source, n))
    overlap = sum((candidate_grams & source_grams).values())
    return overlap / sum(candidate_grams.values())


def support_score(answer: str, context: str, n: int = 4) -> dict[str, float]:
    """Automatic support proxy for one answer against its retrieved context.

    Returns the mean and minimum per-sentence containment. The minimum is the
    more diagnostic of the two: a single unsupported sentence appended to three
    well-grounded ones is exactly the failure mode of interest, and a mean
    hides it.
    """
    sentences = split_sentences(answer)
    if not sentences or not context.strip():
        return {
            "support_mean": 0.0,
            "support_min": 0.0,
            "n_sentences": float(len(sentences)),
            "n_unsupported_sentences": float(len(sentences)),
        }

    scores = [_ngram_containment(s, context, n) for s in sentences]
    threshold = 0.5
    return {
        "support_mean": sum(scores) / len(scores),
        "support_min": min(scores),
        "n_sentences": float(len(sentences)),
        "n_unsupported_sentences": float(sum(1 for s in scores if s < threshold)),
    }


@dataclass
class AbstentionResult:
    """Behaviour on the unanswerable tier of the benchmark."""

    n_unanswerable: int
    n_abstained: int
    n_answered: int
    n_answerable: int
    n_false_abstentions: int

    @property
    def abstention_recall(self) -> float:
        """Of the questions that should be declined, how many were."""
        return self.n_abstained / self.n_unanswerable if self.n_unanswerable else float("nan")

    @property
    def false_abstention_rate(self) -> float:
        """Of the answerable questions, how many were wrongly declined.

        Reported alongside abstention recall because a system that declines
        everything scores a perfect recall and is useless.
        """
        return (
            self.n_false_abstentions / self.n_answerable
            if self.n_answerable
            else float("nan")
        )

    def to_dict(self) -> dict:
        return {
            "n_unanswerable": self.n_unanswerable,
            "n_abstained": self.n_abstained,
            "n_answered_despite_unanswerable": self.n_answered,
            "abstention_recall": round(self.abstention_recall, 4),
            "n_answerable": self.n_answerable,
            "n_false_abstentions": self.n_false_abstentions,
            "false_abstention_rate": round(self.false_abstention_rate, 4),
        }


def score_abstention(records: Sequence[dict]) -> AbstentionResult:
    """Compute abstention behaviour from per-question result records.

    Each record needs ``answerable`` (from the benchmark) and ``status`` (from
    the pipeline).
    """
    unanswerable = [r for r in records if not r.get("answerable", True)]
    answerable = [r for r in records if r.get("answerable", True)]

    abstained = sum(1 for r in unanswerable if r.get("status") == "abstained")
    false_abstentions = sum(1 for r in answerable if r.get("status") == "abstained")

    return AbstentionResult(
        n_unanswerable=len(unanswerable),
        n_abstained=abstained,
        n_answered=len(unanswerable) - abstained,
        n_answerable=len(answerable),
        n_false_abstentions=false_abstentions,
    )


@dataclass
class ClaimJudgement:
    """One annotator's verdict on one claim extracted from one answer."""

    question_id: str
    condition: str
    claim_index: int
    annotator: str
    verdict: str
    """supported | contradicted | not_in_context | unverifiable"""
    cited_chunk_id: str | None = None
    notes: str = ""

    VERDICTS = ("supported", "contradicted", "not_in_context", "unverifiable")

    def __post_init__(self) -> None:
        if self.verdict not in self.VERDICTS:
            raise ValueError(
                f"verdict must be one of {self.VERDICTS}, got {self.verdict!r}"
            )


def aggregate_human_judgements(judgements: Sequence[ClaimJudgement]) -> dict:
    """Per-condition faithfulness rates from annotator verdicts.

    ``hallucination_rate`` is the fraction of claims that are either
    contradicted by the retrieved context or absent from it. That is the
    quantity the paper's hallucination claim needs, and it is defined here
    once so that the number in the text and the number in the code cannot
    diverge.
    """
    by_condition: dict[str, Counter] = {}
    for judgement in judgements:
        by_condition.setdefault(judgement.condition, Counter())[judgement.verdict] += 1

    out: dict[str, dict] = {}
    for condition, counts in by_condition.items():
        total = sum(counts.values())
        if not total:
            continue
        unsupported = counts["contradicted"] + counts["not_in_context"]
        out[condition] = {
            "n_claims": total,
            "supported_rate": round(counts["supported"] / total, 4),
            "contradicted_rate": round(counts["contradicted"] / total, 4),
            "not_in_context_rate": round(counts["not_in_context"] / total, 4),
            "unverifiable_rate": round(counts["unverifiable"] / total, 4),
            "hallucination_rate": round(unsupported / total, 4),
        }
    return out


def krippendorff_alpha_nominal(
    judgements: Sequence[ClaimJudgement],
) -> float:
    """Inter-annotator agreement on nominal verdicts.

    Reported because a hallucination rate derived from unreliable annotation
    is not evidence. Krippendorff's alpha is used rather than Cohen's kappa
    because it tolerates more than two annotators and missing judgements,
    which is the realistic situation for a small annotation effort.
    """
    units: dict[tuple[str, str, int], list[str]] = {}
    for judgement in judgements:
        key = (judgement.question_id, judgement.condition, judgement.claim_index)
        units.setdefault(key, []).append(judgement.verdict)

    usable = {k: v for k, v in units.items() if len(v) >= 2}
    if not usable:
        return float("nan")

    observed_numerator = 0.0
    observed_denominator = 0.0
    all_values: Counter = Counter()

    for values in usable.values():
        m = len(values)
        all_values.update(values)
        counts = Counter(values)
        agreeing = sum(c * (c - 1) for c in counts.values())
        observed_numerator += agreeing / (m - 1)
        observed_denominator += m

    if observed_denominator == 0:
        return float("nan")

    observed_agreement = observed_numerator / observed_denominator

    total = sum(all_values.values())
    if total <= 1:
        return float("nan")
    expected_agreement = sum(
        count * (count - 1) for count in all_values.values()
    ) / (total * (total - 1))

    if expected_agreement >= 1.0:
        return float("nan")
    return (observed_agreement - expected_agreement) / (1 - expected_agreement)
