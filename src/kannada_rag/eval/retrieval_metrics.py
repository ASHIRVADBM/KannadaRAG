"""Retrieval-quality metrics against manually identified relevant passages.

Retrieval is the component that distinguishes this system from a prompted LLM,
so it has to be measured on its own terms rather than inferred from
end-to-end answer quality. A high BERTScore is compatible with retrieval
having failed entirely -- a fluent model can produce a plausible Kannada
sentence from a weak context -- so answer quality cannot stand in for
retrieval quality.

Each benchmark question carries a set of ``gold_chunk_ids`` (or, more robustly
across re-chunkings, ``gold_pages``): the passages a human annotator judged to
contain the evidence needed to answer it. All metrics here are computed
against those judgements over the **unfiltered** ranked candidate list, so
they measure the index and encoder rather than the abstention threshold.

Implemented:

``recall_at_k``      fraction of gold passages retrieved in the top k
``precision_at_k``   fraction of the top k that are gold
``hit_rate_at_k``    whether *any* gold passage appears in the top k
``mrr``              reciprocal rank of the first gold passage
``ndcg_at_k``        normalised discounted cumulative gain, binary relevance
``evidence_hit``     whether the passages actually shown to the generator
                     (post-threshold) contain at least one gold passage
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "RetrievalScores",
    "recall_at_k",
    "precision_at_k",
    "hit_rate_at_k",
    "mrr",
    "ndcg_at_k",
    "score_retrieval",
]


@dataclass
class RetrievalScores:
    """Retrieval metrics for one question at one cut-off."""

    k: int
    recall: float
    precision: float
    hit: float
    reciprocal_rank: float
    ndcg: float
    n_gold: int
    n_retrieved: int

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}recall@{self.k}": self.recall,
            f"{prefix}precision@{self.k}": self.precision,
            f"{prefix}hit@{self.k}": self.hit,
            f"{prefix}mrr@{self.k}": self.reciprocal_rank,
            f"{prefix}ndcg@{self.k}": self.ndcg,
        }


def _as_set(values: Iterable[str] | None) -> set[str]:
    return set(values or ())


def recall_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = _as_set(gold)
    if not gold_set:
        return float("nan")
    return len(gold_set & set(retrieved[:k])) / len(gold_set)


def precision_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    window = retrieved[:k]
    if not window:
        return 0.0
    return len(_as_set(gold) & set(window)) / len(window)


def hit_rate_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    return 1.0 if _as_set(gold) & set(retrieved[:k]) else 0.0


def mrr(retrieved: Sequence[str], gold: Iterable[str], k: int | None = None) -> float:
    """Reciprocal rank of the first relevant item; 0 if none in the window."""
    gold_set = _as_set(gold)
    window = retrieved[:k] if k else retrieved
    for rank, item in enumerate(window, start=1):
        if item in gold_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], gold: Iterable[str], k: int) -> float:
    """nDCG with binary relevance and log2 discount.

    The ideal ranking places every gold passage at the top, so the ideal DCG
    is the sum of the discount over ``min(len(gold), k)`` positions.
    """
    gold_set = _as_set(gold)
    if not gold_set:
        return float("nan")

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in gold_set
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(gold_set), k) + 1)
    )
    return dcg / ideal if ideal > 0 else 0.0


def score_retrieval(
    retrieved_ids: Sequence[str],
    gold_ids: Iterable[str],
    k: int,
) -> RetrievalScores:
    gold_set = _as_set(gold_ids)
    return RetrievalScores(
        k=k,
        recall=recall_at_k(retrieved_ids, gold_set, k),
        precision=precision_at_k(retrieved_ids, gold_set, k),
        hit=hit_rate_at_k(retrieved_ids, gold_set, k),
        reciprocal_rank=mrr(retrieved_ids, gold_set, k),
        ndcg=ndcg_at_k(retrieved_ids, gold_set, k),
        n_gold=len(gold_set),
        n_retrieved=len(retrieved_ids[:k]),
    )


def score_retrieval_multi_k(
    retrieved_ids: Sequence[str],
    gold_ids: Iterable[str],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """All metrics at several cut-offs, flattened for the results log.

    Reporting a curve over k rather than a single value is what lets a reader
    tell a retrieval failure (nothing relevant at any k) from a ranking
    failure (relevant passages present but ranked low).
    """
    out: dict[str, float] = {}
    for k in ks:
        out.update(score_retrieval(retrieved_ids, gold_ids, k).to_dict())
    return out


def evidence_hit(selected_ids: Sequence[str], gold_ids: Iterable[str]) -> float:
    """Did the generator actually see at least one gold passage?

    Computed over the post-threshold selection, unlike every metric above.
    This is the number that bounds how often a grounded answer *could* have
    been correct: when it is 0, any correct-looking answer came from the
    model's parametric memory, not from the corpus.
    """
    return 1.0 if _as_set(gold_ids) & set(selected_ids) else 0.0
