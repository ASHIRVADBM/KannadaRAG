"""Retrieval with a formally specified selection rule.

The number of passages passed to the generator is decided by a deterministic
rule, not by a human judgement of "query complexity". The rule is:

    candidates = index.search(q, k_max)
    top       = max(score for candidates)                       # highest similarity
    kept      = [c for c in candidates
                 if c.score >= sim_threshold                    # absolute floor
                 and c.score >= relative_floor * top]           # relative floor
    kept      = kept[:k_max]

with abstention when ``kept`` is empty.

Both floors do distinct work. The **absolute floor** decides whether the corpus
contains anything relevant at all; it is what allows the system to say "this is
not in the corpus" instead of generating from weak context. The **relative
floor** decides how many of the retrieved passages are comparably relevant to
the best one; a query whose top hit dominates receives a small, focused
context, while a query whose top several hits score similarly receives a wider
one. This reproduces the intent of the earlier hand-tuned "K = 3 or 5" policy
while being fully determined by the configuration.

Both thresholds are hyperparameters and are tuned on a development split that
is disjoint from the evaluation questions. ``python -m kannada_rag.cli ablate threshold``
performs that calibration and writes the resulting sweep, so the chosen value
is reported as a calibration result rather than a magic number.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import RetrievalConfig
from .index import Embedder, ScoredChunk, VectorIndex
from .text import normalise_for_index

logger = logging.getLogger(__name__)

__all__ = ["RetrievalResult", "Retriever"]


@dataclass
class RetrievalResult:
    """Everything the evaluation harness needs about one retrieval call.

    ``candidates`` holds the full unfiltered ranked list. Retrieval metrics
    (Recall@K, MRR, nDCG) are computed over ``candidates``; the generator sees
    only ``selected``. Logging both means retrieval quality can be measured
    independently of the threshold that happens to be configured.
    """

    query: str
    normalised_query: str
    candidates: list[ScoredChunk] = field(default_factory=list)
    selected: list[ScoredChunk] = field(default_factory=list)
    abstained: bool = False
    abstain_reason: str | None = None

    @property
    def top_score(self) -> float:
        return self.candidates[0].score if self.candidates else float("nan")

    @property
    def k_effective(self) -> int:
        return len(self.selected)

    @property
    def context(self) -> str:
        """The passage text handed to the generator, with citations attached."""
        return "\n\n".join(
            f"[{i + 1}] ({sc.chunk.citation}) {sc.chunk.text}"
            for i, sc in enumerate(self.selected)
        )

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "top_score": None if self.candidates == [] else round(self.top_score, 6),
            "k_effective": self.k_effective,
            "selected": [sc.to_dict() for sc in self.selected],
            "candidates": [
                {"chunk_id": sc.chunk.chunk_id, "score": round(sc.score, 6), "rank": sc.rank}
                for sc in self.candidates
            ],
        }


class Retriever:
    """Dense retriever applying the absolute/relative selection rule."""

    def __init__(self, embedder: Embedder, index: VectorIndex, cfg: RetrievalConfig):
        self.embedder = embedder
        self.index = index
        self.cfg = cfg

    def retrieve(self, query: str, k_max: int | None = None) -> RetrievalResult:
        k_max = k_max or self.cfg.k_max
        normalised = normalise_for_index(query)

        result = RetrievalResult(query=query, normalised_query=normalised)

        vector = self.embedder.encode([normalised])
        result.candidates = self.index.search(vector, k_max)

        if not result.candidates:
            result.abstained = True
            result.abstain_reason = "empty_index"
            return result

        top = result.candidates[0].score
        absolute_floor = self.cfg.sim_threshold
        relative_floor = self.cfg.relative_floor * top

        kept = [
            sc
            for sc in result.candidates
            if sc.score >= absolute_floor and sc.score >= relative_floor
        ]

        if not kept:
            if self.cfg.abstain_when_empty:
                result.abstained = True
                result.abstain_reason = "below_similarity_threshold"
                return result
            kept = result.candidates[: self.cfg.k_min]

        result.selected = kept[: self.cfg.k_max]
        return result

    def retrieve_candidates_only(self, query: str, k: int) -> list[ScoredChunk]:
        """Unfiltered top-k, used by the retrieval-metric evaluator.

        Recall@K must be measured against the ranked list the index produces,
        independently of the abstention threshold, otherwise the threshold and
        the retrieval quality become impossible to disentangle.
        """
        vector = self.embedder.encode([normalise_for_index(query)])
        return self.index.search(vector, k)
