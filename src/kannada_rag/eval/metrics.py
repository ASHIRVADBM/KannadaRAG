"""Response-quality metrics, implemented to their published definitions.

Every metric here computes what its name says. That sounds like a low bar, but
the previous version of this work reported a "METEOR" that was only the
harmonic-mean term with the fragmentation penalty omitted, and a "ROUGE-2"
that appeared in a results table without ever being defined. A metric that
omits part of its definition is a different metric and must not carry the
original name.

What each function computes:

``rouge_n``
    F1 over matched n-grams, with *clipped* counts (an n-gram occurring twice
    in the hypothesis and once in the reference matches once, not twice).
    n = 1 and n = 2 give ROUGE-1 and ROUGE-2.

``rouge_l``
    F1 over the longest common subsequence of the two token sequences.

``meteor``
    The full METEOR: unigram alignment, the recall-weighted harmonic mean
    ``F_mean = 10PR / (R + 9P)``, **and** the fragmentation penalty
    ``gamma * (chunks / matches) ^ beta``. Score = ``F_mean * (1 - penalty)``.
    Because no Kannada stemmer or WordNet exists, only the exact-match
    alignment stage runs; the optional character-n-gram stage is available via
    ``fuzzy_threshold`` and is reported separately when used, never silently
    folded into a score labelled METEOR.

``chrf``
    Character n-gram F-score. Included because it is the metric best suited to
    an agglutinative script: two inflected forms of the same lemma share most
    character n-grams and no word unigrams. Reported alongside ROUGE rather
    than instead of it.

``bert_score``
    Token-level greedy cosine matching over contextual embeddings, with the
    checkpoint, layer and IDF setting all pinned in :class:`MetricConfig` and
    recorded in every output file. A BERTScore is not reproducible without
    those three facts.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Sequence

from ..text import character_ngrams, tokenise_for_metrics

__all__ = [
    "MetricConfig",
    "rouge_n",
    "rouge_l",
    "meteor",
    "chrf",
    "BERTScorer",
    "score_pair",
]


@dataclass(frozen=True)
class MetricConfig:
    """Everything needed to reproduce a metric value exactly."""

    # --- BERTScore: all three fields are required for reproducibility --------
    bertscore_model: str = "xlm-roberta-base"
    bertscore_revision: str | None = None
    bertscore_layer: int = 9
    """Layer from which representations are taken. The BERTScore authors
    select this per model on WMT correlation data; for xlm-roberta-base the
    published default is layer 9. Reporting a BERTScore without the layer
    makes it unreproducible."""
    bertscore_idf: bool = False
    """IDF weighting. Disabled by default because the 25-question reference set
    is far too small to estimate document frequencies stably; enabling it on a
    corpus this size adds variance rather than signal."""
    bertscore_batch_size: int = 16

    # --- METEOR parameters (Banerjee & Lavie 2005 defaults) -----------------
    meteor_alpha: float = 0.9    # recall weight inside F_mean
    meteor_beta: float = 3.0     # penalty exponent
    meteor_gamma: float = 0.5    # penalty weight
    meteor_fuzzy_threshold: float | None = None
    """If set, a character-n-gram similarity above this value counts as a
    match in a second alignment stage. Off by default; when on, the result is
    labelled ``meteor_fuzzy`` and never ``meteor``."""

    # --- chrF ---------------------------------------------------------------
    chrf_n: int = 6
    chrf_beta: float = 2.0

    keep_numerals: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- ROUGE


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _prf(matches: int, hyp_total: int, ref_total: int) -> dict[str, float]:
    precision = matches / hyp_total if hyp_total else 0.0
    recall = matches / ref_total if ref_total else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_n(hypothesis: str, reference: str, n: int = 1, keep_numerals: bool = True) -> dict[str, float]:
    """ROUGE-N with clipped n-gram counts."""
    hyp = tokenise_for_metrics(hypothesis, keep_numerals)
    ref = tokenise_for_metrics(reference, keep_numerals)

    hyp_ngrams = _ngrams(hyp, n)
    ref_ngrams = _ngrams(ref, n)

    matches = sum((hyp_ngrams & ref_ngrams).values())
    return _prf(matches, sum(hyp_ngrams.values()), sum(ref_ngrams.values()))


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence, O(len(a) * len(b)) time and
    O(min) space."""
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a

    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b):
            if token_a == token_b:
                current.append(previous[j] + 1)
            else:
                current.append(max(current[j], previous[j + 1]))
        previous = current
    return previous[-1]


def rouge_l(hypothesis: str, reference: str, keep_numerals: bool = True) -> dict[str, float]:
    """ROUGE-L: F1 over the longest common subsequence."""
    hyp = tokenise_for_metrics(hypothesis, keep_numerals)
    ref = tokenise_for_metrics(reference, keep_numerals)
    lcs = _lcs_length(hyp, ref)
    return _prf(lcs, len(hyp), len(ref))


# -------------------------------------------------------------------- METEOR


def _align_exact(hyp: Sequence[str], ref: Sequence[str]) -> list[tuple[int, int]]:
    """Greedy left-to-right exact unigram alignment.

    Each reference token may be used at most once. Returns (hyp_idx, ref_idx)
    pairs sorted by hypothesis position, which is what the chunk count needs.
    """
    used_ref: set[int] = set()
    alignment: list[tuple[int, int]] = []

    for h_idx, h_token in enumerate(hyp):
        for r_idx, r_token in enumerate(ref):
            if r_idx in used_ref:
                continue
            if h_token == r_token:
                alignment.append((h_idx, r_idx))
                used_ref.add(r_idx)
                break
    return alignment


def _align_fuzzy(
    hyp: Sequence[str],
    ref: Sequence[str],
    aligned: list[tuple[int, int]],
    threshold: float,
    n: int = 3,
) -> list[tuple[int, int]]:
    """Second alignment stage over character n-gram similarity.

    Stands in for the stem and synonym stages of the original METEOR, which
    require resources that do not exist for Kannada. Only ever used when
    explicitly requested, and the result is reported under a different name.
    """
    used_hyp = {h for h, _ in aligned}
    used_ref = {r for _, r in aligned}
    extra: list[tuple[int, int]] = []

    for h_idx, h_token in enumerate(hyp):
        if h_idx in used_hyp:
            continue
        best, best_sim = None, 0.0
        h_grams = set(character_ngrams(h_token, n))
        if not h_grams:
            continue
        for r_idx, r_token in enumerate(ref):
            if r_idx in used_ref:
                continue
            r_grams = set(character_ngrams(r_token, n))
            if not r_grams:
                continue
            sim = len(h_grams & r_grams) / len(h_grams | r_grams)
            if sim > best_sim:
                best, best_sim = r_idx, sim
        if best is not None and best_sim >= threshold:
            extra.append((h_idx, best))
            used_ref.add(best)
            used_hyp.add(h_idx)

    return sorted(aligned + extra)


def _count_chunks(alignment: list[tuple[int, int]]) -> int:
    """Number of contiguous runs in the alignment.

    A chunk is a maximal sequence of aligned pairs that is adjacent in *both*
    the hypothesis and the reference. Fewer chunks means better-preserved word
    order; the penalty grows as the alignment fragments.
    """
    if not alignment:
        return 0
    ordered = sorted(alignment)
    chunks = 1
    for (h_prev, r_prev), (h_cur, r_cur) in zip(ordered, ordered[1:]):
        if not (h_cur == h_prev + 1 and r_cur == r_prev + 1):
            chunks += 1
    return chunks


def meteor(
    hypothesis: str,
    reference: str,
    cfg: MetricConfig | None = None,
) -> dict[str, float]:
    """METEOR, complete with the fragmentation penalty.

    Returns the components as well as the final score, so that a reader can
    verify the arithmetic and see how much of any difference between systems
    comes from alignment quality versus word order.
    """
    cfg = cfg or MetricConfig()
    hyp = tokenise_for_metrics(hypothesis, cfg.keep_numerals)
    ref = tokenise_for_metrics(reference, cfg.keep_numerals)

    if not hyp or not ref:
        return {
            "meteor": 0.0, "precision": 0.0, "recall": 0.0,
            "f_mean": 0.0, "penalty": 0.0, "matches": 0, "chunks": 0,
        }

    alignment = _align_exact(hyp, ref)
    if cfg.meteor_fuzzy_threshold is not None:
        alignment = _align_fuzzy(hyp, ref, alignment, cfg.meteor_fuzzy_threshold)

    matches = len(alignment)
    if matches == 0:
        return {
            "meteor": 0.0, "precision": 0.0, "recall": 0.0,
            "f_mean": 0.0, "penalty": 0.0, "matches": 0, "chunks": 0,
        }

    precision = matches / len(hyp)
    recall = matches / len(ref)

    alpha = cfg.meteor_alpha
    denominator = alpha * precision + (1 - alpha) * recall
    f_mean = (precision * recall) / denominator if denominator > 0 else 0.0

    chunks = _count_chunks(alignment)
    penalty = cfg.meteor_gamma * (chunks / matches) ** cfg.meteor_beta

    return {
        "meteor": f_mean * (1 - penalty),
        "precision": precision,
        "recall": recall,
        "f_mean": f_mean,
        "penalty": penalty,
        "matches": matches,
        "chunks": chunks,
    }


# ----------------------------------------------------------------------- chrF


def chrf(hypothesis: str, reference: str, cfg: MetricConfig | None = None) -> dict[str, float]:
    """Character n-gram F-score, averaged over orders 1..n.

    Morphologically robust: inflected variants of a Kannada lemma share
    character n-grams even when they share no whole-word token. This is the
    metric to lead with for a language where surface overlap understates
    agreement.
    """
    cfg = cfg or MetricConfig()
    precisions, recalls = [], []

    for order in range(1, cfg.chrf_n + 1):
        hyp_grams = Counter(character_ngrams(hypothesis, order))
        ref_grams = Counter(character_ngrams(reference, order))
        if not hyp_grams or not ref_grams:
            continue
        overlap = sum((hyp_grams & ref_grams).values())
        precisions.append(overlap / sum(hyp_grams.values()))
        recalls.append(overlap / sum(ref_grams.values()))

    if not precisions:
        return {"chrf": 0.0, "precision": 0.0, "recall": 0.0}

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)

    beta_sq = cfg.chrf_beta ** 2
    denominator = beta_sq * precision + recall
    score = (
        (1 + beta_sq) * precision * recall / denominator if denominator > 0 else 0.0
    )
    return {"chrf": score, "precision": precision, "recall": recall}


# ------------------------------------------------------------------ BERTScore


class BERTScorer:
    """BERTScore with an explicitly pinned checkpoint, layer and IDF setting.

    Loaded lazily and reused across a whole evaluation run: instantiating the
    encoder per question dominates the runtime and, with IDF enabled, would
    change the statistics between questions.
    """

    def __init__(self, cfg: MetricConfig | None = None):
        self.cfg = cfg or MetricConfig()
        self._scorer = None

    def _ensure(self):
        if self._scorer is None:
            from bert_score import BERTScorer as _BS

            self._scorer = _BS(
                model_type=self.cfg.bertscore_model,
                num_layers=self.cfg.bertscore_layer,
                idf=self.cfg.bertscore_idf,
                batch_size=self.cfg.bertscore_batch_size,
                lang="kn",
                rescale_with_baseline=False,
            )
        return self._scorer

    def score(
        self, hypotheses: Sequence[str], references: Sequence[str]
    ) -> list[dict[str, float]]:
        scorer = self._ensure()
        precision, recall, f1 = scorer.score(list(hypotheses), list(references))
        return [
            {
                "bert_precision": float(p),
                "bert_recall": float(r),
                "bert_f1": float(f),
            }
            for p, r, f in zip(precision, recall, f1)
        ]

    def provenance(self) -> dict:
        """Recorded in every results file so the score can be reproduced."""
        return {
            "model": self.cfg.bertscore_model,
            "revision": self.cfg.bertscore_revision,
            "layer": self.cfg.bertscore_layer,
            "idf": self.cfg.bertscore_idf,
            "rescale_with_baseline": False,
        }


# ------------------------------------------------------------------ aggregate


def score_pair(
    hypothesis: str, reference: str, cfg: MetricConfig | None = None
) -> dict[str, float]:
    """All surface metrics for one (hypothesis, reference) pair.

    BERTScore is excluded here because it is computed in batch over the whole
    run for efficiency; :mod:`kannada_rag.eval.runner` merges it in.
    """
    cfg = cfg or MetricConfig()
    r1 = rouge_n(hypothesis, reference, 1, cfg.keep_numerals)
    r2 = rouge_n(hypothesis, reference, 2, cfg.keep_numerals)
    rl = rouge_l(hypothesis, reference, cfg.keep_numerals)
    mt = meteor(hypothesis, reference, cfg)
    cf = chrf(hypothesis, reference, cfg)

    return {
        "rouge1_f1": r1["f1"],
        "rouge1_precision": r1["precision"],
        "rouge1_recall": r1["recall"],
        "rouge2_f1": r2["f1"],
        "rougeL_f1": rl["f1"],
        "rougeL_precision": rl["precision"],
        "rougeL_recall": rl["recall"],
        "meteor": mt["meteor"],
        "meteor_f_mean": mt["f_mean"],
        "meteor_penalty": mt["penalty"],
        "meteor_chunks": float(mt["chunks"]),
        "chrf": cf["chrf"],
        "hyp_tokens": float(len(tokenise_for_metrics(hypothesis, cfg.keep_numerals))),
        "ref_tokens": float(len(tokenise_for_metrics(reference, cfg.keep_numerals))),
    }
