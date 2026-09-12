"""Design-choice sweeps.

Every configuration constant in this system was chosen for a reason, and until
those reasons are measured they are assertions. This module sweeps the four
that matter and writes the result as a table, so the paper can say which
choices contribute to performance and by how much instead of describing them
as self-evidently correct.

The sweeps:

``chunk_size`` / ``chunk_overlap``
    Changing either invalidates the index, so each setting rebuilds it. This
    is the expensive sweep and the one that answers whether 500/50 is a good
    operating point or merely the first one tried.

``k_max`` / ``sim_threshold`` / ``relative_floor``
    Retrieval-only sweeps. They share one index and are consequently cheap, so
    they are evaluated on retrieval metrics alone over a dense grid, with no
    generation at all. This is also how the threshold is calibrated: on a
    development split disjoint from the evaluation questions.

``embedding model``
    The claim that a multilingual checkpoint "directly reflects Kannada
    semantics" is only supportable by comparison. This sweep runs the same
    benchmark against alternative encoders (LaBSE, MuRIL, multilingual-e5) and
    reports retrieval quality for each.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from itertools import product
from pathlib import Path
from typing import Iterable, Sequence

from ..config import PipelineConfig
from ..index import Embedder
from ..pipeline import build_index
from ..retriever import Retriever
from ..text import normalise_for_index
from .benchmark import Benchmark
from .retrieval_metrics import score_retrieval_multi_k
from .stats import describe
from .tables import Table

logger = logging.getLogger(__name__)

__all__ = [
    "sweep_retrieval",
    "sweep_chunking",
    "sweep_embedding_models",
    "calibrate_threshold",
]

DEFAULT_KS = (1, 3, 5, 10)


def _gold_ids(question, use_pages: bool = True) -> list[str]:
    if use_pages and question.gold_pages and question.source_file:
        return [f"{question.source_file}#p{p}" for p in question.gold_pages]
    return list(question.gold_chunk_ids)


def _retrieved_ids(candidates, use_pages: bool = True) -> list[str]:
    if use_pages:
        return [f"{sc.chunk.source_file}#p{sc.chunk.page}" for sc in candidates]
    return [sc.chunk.chunk_id for sc in candidates]


def sweep_retrieval(
    cfg: PipelineConfig,
    benchmark: Benchmark,
    k_values: Sequence[int] = (1, 3, 5, 10),
    thresholds: Sequence[float] = (0.0, 0.35, 0.45, 0.55, 0.65, 0.75),
    relative_floors: Sequence[float] = (0.0, 0.75, 0.85, 0.95),
    output_dir: str | Path | None = None,
) -> Table:
    """Sweep retrieval hyperparameters over one fixed index.

    No generation runs here, so the whole grid costs one embedding pass over
    the questions. Reporting a grid rather than a single operating point is
    what shows a reader that the threshold was selected rather than assumed.
    """
    index = build_index(cfg)
    embedder = Embedder(cfg.embedding)
    questions = [q for q in benchmark.answerable if _gold_ids(q)]

    if not questions:
        raise ValueError(
            "No questions carry gold evidence; retrieval cannot be evaluated. "
            "Annotate gold_pages in the benchmark first."
        )

    rows: list[list] = []
    records: list[dict] = []
    max_k = max(k_values)

    # Retrieve once at the widest k, then re-filter in memory for every cell.
    cached: dict[str, list] = {}
    for question in questions:
        vector = embedder.encode([normalise_for_index(question.question)])
        cached[question.question_id] = index.search(vector, max_k)

    for k, threshold, floor in product(k_values, thresholds, relative_floors):
        per_question: dict[str, dict[str, float]] = {}
        k_effective: list[float] = []
        abstentions = 0

        for question in questions:
            candidates = cached[question.question_id][:k]
            gold = _gold_ids(question)
            scores = score_retrieval_multi_k(_retrieved_ids(candidates), gold, (k,))
            per_question[question.question_id] = scores

            top = candidates[0].score if candidates else 0.0
            selected = [
                sc for sc in candidates
                if sc.score >= threshold and sc.score >= floor * top
            ]
            if not selected:
                abstentions += 1
            k_effective.append(len(selected))

        recall = [v[f"recall@{k}"] for v in per_question.values()]
        precision = [v[f"precision@{k}"] for v in per_question.values()]
        ndcg = [v[f"ndcg@{k}"] for v in per_question.values()]

        row = [
            k,
            threshold,
            floor,
            describe(recall)["mean"],
            describe(precision)["mean"],
            describe(ndcg)["mean"],
            round(sum(k_effective) / len(k_effective), 2),
            round(abstentions / len(questions), 3),
        ]
        rows.append(row)
        records.append(
            {
                "k_max": k, "sim_threshold": threshold, "relative_floor": floor,
                "recall": describe(recall), "precision": describe(precision),
                "ndcg": describe(ndcg),
                "mean_k_effective": row[6], "abstention_rate": row[7],
            }
        )

    table = Table(
        caption=(
            "Retrieval ablation over candidate depth, absolute similarity threshold "
            "and relative floor, on the annotated development questions. "
            "Abstention rate is the fraction of answerable questions for which no "
            "passage cleared both floors."
        ),
        label="tab:ablation-retrieval",
        columns=[
            "k", "threshold", "rel. floor", "Recall@k", "Precision@k",
            "nDCG@k", "mean k_eff", "abstention rate",
        ],
        rows=rows,
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / "ablation_retrieval.csv")
        (output_dir / "ablation_retrieval.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return table


def calibrate_threshold(
    cfg: PipelineConfig,
    dev_benchmark: Benchmark,
    thresholds: Sequence[float] = tuple(round(0.05 * i, 2) for i in range(0, 20)),
    output_dir: str | Path | None = None,
) -> dict:
    """Select the similarity threshold on a development split.

    The objective balances two error types: answering from irrelevant context
    (threshold too low) and declining a question the corpus can answer
    (threshold too high). The chosen value maximises the F1 of the "should
    answer" decision on the development set, which requires the development
    set to contain unanswerable questions -- which is why the benchmark has an
    unanswerable tier.

    Critically, this runs on questions **disjoint from the evaluation set**. A
    threshold tuned on the test questions is a form of leakage, and the earlier
    version of this work could not demonstrate otherwise.
    """
    index = build_index(cfg)
    embedder = Embedder(cfg.embedding)
    questions = list(dev_benchmark)

    if not any(not q.answerable for q in questions):
        raise ValueError(
            "Threshold calibration needs unanswerable questions in the development "
            "split; otherwise the 'should abstain' decision has no negative class."
        )

    top_scores: dict[str, float] = {}
    for question in questions:
        vector = embedder.encode([normalise_for_index(question.question)])
        candidates = index.search(vector, cfg.retrieval.k_max)
        top_scores[question.question_id] = candidates[0].score if candidates else 0.0

    results = []
    for threshold in thresholds:
        tp = fp = fn = tn = 0
        for question in questions:
            answered = top_scores[question.question_id] >= threshold
            if question.answerable and answered:
                tp += 1
            elif question.answerable and not answered:
                fn += 1
            elif not question.answerable and answered:
                fp += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        results.append(
            {
                "threshold": threshold, "precision": round(precision, 4),
                "recall": round(recall, 4), "f1": round(f1, 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            }
        )

    best = max(results, key=lambda r: r["f1"])
    payload = {
        "selected_threshold": best["threshold"],
        "selected_f1": best["f1"],
        "n_dev_questions": len(questions),
        "n_unanswerable": sum(1 for q in questions if not q.answerable),
        "sweep": results,
        "note": (
            "Calibrated on the development split only. The evaluation questions "
            "took no part in selecting this value."
        ),
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "threshold_calibration.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    logger.info("Selected similarity threshold %.2f (F1 %.3f)", best["threshold"], best["f1"])
    return payload


def sweep_chunking(
    cfg: PipelineConfig,
    benchmark: Benchmark,
    chunk_sizes: Sequence[int] = (250, 500, 750, 1000),
    overlaps: Sequence[int] = (0, 50, 100, 200),
    output_dir: str | Path | None = None,
) -> Table:
    """Sweep chunk size and overlap, rebuilding the index for each setting.

    Evaluated on retrieval metrics only. Generation is held out of this sweep
    deliberately: adding it would multiply the cost by the number of models
    without changing which chunking setting retrieves the evidence best.
    """
    questions = [q for q in benchmark.answerable if _gold_ids(q)]
    if not questions:
        raise ValueError("No gold evidence annotated; cannot evaluate chunking.")

    rows: list[list] = []
    records: list[dict] = []

    for size, overlap in product(chunk_sizes, overlaps):
        if overlap >= size:
            continue

        variant = cfg.with_(chunk=replace(cfg.chunk, chunk_size=size, chunk_overlap=overlap))
        variant = variant.with_(index_dir=Path(cfg.index_dir) / f"sweep_c{size}_o{overlap}")

        logger.info("Building index for chunk_size=%d overlap=%d", size, overlap)
        index = build_index(variant, force=False)
        embedder = Embedder(variant.embedding)
        retriever = Retriever(embedder, index, variant.retrieval)

        recalls, precisions, ndcgs, hits = [], [], [], []
        for question in questions:
            candidates = retriever.retrieve_candidates_only(question.question, 10)
            scores = score_retrieval_multi_k(
                _retrieved_ids(candidates), _gold_ids(question), (5,)
            )
            recalls.append(scores["recall@5"])
            precisions.append(scores["precision@5"])
            ndcgs.append(scores["ndcg@5"])
            hits.append(scores["hit@5"])

        row = [
            size, overlap, len(index),
            describe(recalls)["mean"], describe(precisions)["mean"],
            describe(ndcgs)["mean"], describe(hits)["mean"],
        ]
        rows.append(row)
        records.append(
            {
                "chunk_size": size, "chunk_overlap": overlap, "n_chunks": len(index),
                "recall@5": describe(recalls), "precision@5": describe(precisions),
                "ndcg@5": describe(ndcgs), "hit@5": describe(hits),
            }
        )

    table = Table(
        caption=(
            "Chunking ablation. Each row rebuilds the index at the given chunk size "
            "and overlap and re-evaluates retrieval against the annotated gold "
            "evidence."
        ),
        label="tab:ablation-chunking",
        columns=[
            "Chunk size", "Overlap", "n chunks",
            "Recall@5", "Precision@5", "nDCG@5", "Hit@5",
        ],
        rows=rows,
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / "ablation_chunking.csv")
        (output_dir / "ablation_chunking.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return table


def sweep_embedding_models(
    cfg: PipelineConfig,
    benchmark: Benchmark,
    models: Iterable[str] = (
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "sentence-transformers/LaBSE",
        "intfloat/multilingual-e5-base",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
    ),
    output_dir: str | Path | None = None,
) -> Table:
    """Compare retrieval encoders on annotated Kannada questions.

    This is the empirical comparison that a claim about an encoder's Kannada
    capability requires. Multilingual coverage, contrastive sentence training
    and demonstrated Kannada retrieval performance are three different things,
    and only the third is evidence for the choice made here.
    """
    questions = [q for q in benchmark.answerable if _gold_ids(q)]
    if not questions:
        raise ValueError("No gold evidence annotated; cannot compare encoders.")

    rows: list[list] = []
    records: list[dict] = []

    for model_name in models:
        variant = cfg.with_(embedding=replace(cfg.embedding, model_name=model_name))
        slug = model_name.replace("/", "__")
        variant = variant.with_(index_dir=Path(cfg.index_dir) / f"embed_{slug}")

        try:
            logger.info("Evaluating encoder %s", model_name)
            index = build_index(variant)
            embedder = Embedder(variant.embedding)
            retriever = Retriever(embedder, index, variant.retrieval)
        except Exception as exc:  # noqa: BLE001
            logger.error("Skipping %s: %s", model_name, exc)
            rows.append([model_name, "--", "--", "--", "--", f"failed: {exc}"])
            continue

        recalls, ndcgs, mrrs = [], [], []
        for question in questions:
            candidates = retriever.retrieve_candidates_only(question.question, 10)
            scores = score_retrieval_multi_k(
                _retrieved_ids(candidates), _gold_ids(question), (5, 10)
            )
            recalls.append(scores["recall@5"])
            ndcgs.append(scores["ndcg@10"])
            mrrs.append(scores["mrr@10"])

        row = [
            model_name.split("/")[-1],
            embedder.dim,
            describe(recalls)["mean"],
            describe(ndcgs)["mean"],
            describe(mrrs)["mean"],
            "",
        ]
        rows.append(row)
        records.append(
            {
                "model": model_name, "dim": embedder.dim,
                "recall@5": describe(recalls), "ndcg@10": describe(ndcgs),
                "mrr@10": describe(mrrs),
            }
        )

    table = Table(
        caption=(
            "Retrieval encoder comparison on the annotated Kannada question set. "
            "All settings other than the encoder are held fixed."
        ),
        label="tab:ablation-encoder",
        columns=["Encoder", "dim", "Recall@5", "nDCG@10", "MRR@10", "Note"],
        rows=rows,
    )

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_dir / "ablation_encoder.csv")
        (output_dir / "ablation_encoder.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return table
