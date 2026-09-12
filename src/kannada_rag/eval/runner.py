"""The evaluation runner: one raw log, from which every table is derived.

This module exists to make a specific class of error impossible. Previously,
different tables in the write-up were produced by different ad-hoc scripts at
different times, and consequently disagreed: an overall score that was not the
aggregate of the per-difficulty scores it summarised, and a comparison table
whose rows had been copied from another table. No amount of care in
transcription fixes that; the fix is structural.

Here, a run produces exactly one artefact: a JSONL file with one record per
(question x condition x repeat). Each record contains the question id, the
reference version, the model output, the retrieved passages with scores, the
per-question metric values, and the full configuration that produced it.
:mod:`kannada_rag.eval.tables` then derives *every* published table from that
file by aggregation. An overall mean is the mean of the same records the
per-difficulty means came from, by construction, not by coincidence.

Repeats. ``GenerationConfig.num_repeats`` independent generations per question
support variance reporting. Even at temperature 0.3 the same prompt does not
give the same output twice, and a single-sample evaluation reports one draw
from a distribution as though it were the distribution.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from ..config import PipelineConfig
from ..pipeline import Answer, RAGPipeline
from .benchmark import Benchmark, Question
from .faithfulness import support_score
from .metrics import BERTScorer, MetricConfig, score_pair
from .retrieval_metrics import evidence_hit, score_retrieval_multi_k

logger = logging.getLogger(__name__)

__all__ = ["RunMetadata", "EvaluationRunner", "run_conditions"]

RETRIEVAL_KS = (1, 3, 5, 10)


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return None


@dataclass
class RunMetadata:
    """Environment provenance written as the first line of every log."""

    started_at: str
    git_commit: str | None
    python_version: str
    platform: str
    config: dict
    metric_config: dict
    bertscore_provenance: dict
    benchmark_provenance: dict

    def to_dict(self) -> dict:
        return {"_meta": self.__dict__}


class EvaluationRunner:
    """Runs a benchmark through a pipeline and writes the raw record log."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        metric_config: MetricConfig | None = None,
        bert_scorer: BERTScorer | None = None,
    ):
        self.pipeline = pipeline
        self.metric_config = metric_config or MetricConfig()
        # Shared across conditions so the encoder is loaded once per process.
        self.bert_scorer = bert_scorer or BERTScorer(self.metric_config)

    # ------------------------------------------------------------------ score

    def _best_reference_scores(self, answer_text: str, question: Question) -> dict:
        """Score against every accepted reference, keep the best per metric.

        With multiple valid phrasings, scoring only against the first would
        punish a correct answer for choosing a different but equally correct
        wording -- a systematic penalty in an agglutinative language where the
        same fact admits many surface forms.
        """
        if not question.references:
            return {}

        per_reference = [
            score_pair(answer_text, reference, self.metric_config)
            for reference in question.references
        ]
        best: dict[str, float] = {}
        for scores in per_reference:
            for key, value in scores.items():
                best[key] = max(best.get(key, float("-inf")), value)
        best["n_references"] = float(len(question.references))
        return best

    def _record(
        self,
        question: Question,
        answer: Answer,
        repeat: int,
        cfg: PipelineConfig,
    ) -> dict:
        record: dict = {
            "question_id": question.question_id,
            "question": question.question,
            "difficulty": question.difficulty,
            "answerable": question.answerable,
            "reference_version": question.reference_version,
            "references": question.references,
            "condition": cfg.condition_id,
            "model": cfg.generation.model,
            "provider": cfg.generation.provider,
            "use_retrieval": cfg.use_retrieval,
            "repeat": repeat,
            "status": answer.status,
            "answer": answer.answer,
            "retrieval_latency_s": answer.retrieval_latency_s,
            "generation_latency_s": answer.generation_latency_s,
            "total_latency_s": answer.total_latency_s,
            "is_local": answer.generation.is_local if answer.generation else None,
        }

        # --- response quality ------------------------------------------------
        if question.answerable and answer.answer:
            record.update(self._best_reference_scores(answer.answer, question))

        # --- retrieval quality ----------------------------------------------
        if answer.retrieval is not None:
            candidate_pages = [
                f"{sc.chunk.source_file}#p{sc.chunk.page}" for sc in answer.retrieval.candidates
            ]
            candidate_chunks = [sc.chunk.chunk_id for sc in answer.retrieval.candidates]
            selected_pages = [
                f"{sc.chunk.source_file}#p{sc.chunk.page}" for sc in answer.retrieval.selected
            ]

            gold_pages = [
                f"{question.source_file}#p{p}" for p in question.gold_pages
            ] if question.source_file else []

            if gold_pages:
                record.update(
                    score_retrieval_multi_k(candidate_pages, gold_pages, RETRIEVAL_KS)
                )
                record["evidence_hit"] = evidence_hit(selected_pages, gold_pages)
            elif question.gold_chunk_ids:
                record.update(
                    score_retrieval_multi_k(
                        candidate_chunks, question.gold_chunk_ids, RETRIEVAL_KS
                    )
                )
                record["evidence_hit"] = evidence_hit(
                    [sc.chunk.chunk_id for sc in answer.retrieval.selected],
                    question.gold_chunk_ids,
                )

            record["k_effective"] = answer.retrieval.k_effective
            record["top_score"] = (
                round(answer.retrieval.top_score, 6) if answer.retrieval.candidates else None
            )
            record["retrieved"] = [sc.to_dict() for sc in answer.retrieval.selected]

            # --- faithfulness proxy ------------------------------------------
            if answer.answer and answer.retrieval.selected:
                record.update(support_score(answer.answer, answer.retrieval.context))

        return record

    # -------------------------------------------------------------------- run

    def run(
        self,
        benchmark: Benchmark,
        output_path: str | Path,
        append: bool = False,
    ) -> Path:
        cfg = self.pipeline.cfg
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = RunMetadata(
            started_at=datetime.now(timezone.utc).isoformat(),
            git_commit=_git_commit(),
            python_version=platform.python_version(),
            platform=platform.platform(),
            config=cfg.to_dict(),
            metric_config=self.metric_config.to_dict(),
            bertscore_provenance=self.bert_scorer.provenance(),
            benchmark_provenance=benchmark.provenance(),
        )

        records: list[dict] = []
        n_repeats = max(1, cfg.generation.num_repeats)

        start = time.perf_counter()
        for repeat in range(n_repeats):
            for question in benchmark:
                answer = self.pipeline.answer(question.question)
                records.append(self._record(question, answer, repeat, cfg))
                logger.debug(
                    "%s r%d %s -> %s",
                    question.question_id, repeat, cfg.condition_id, answer.status,
                )
        elapsed = time.perf_counter() - start

        # --- BERTScore in one batch -----------------------------------------
        scorable = [
            r for r in records
            if r.get("answerable") and r.get("answer") and r.get("references")
        ]
        if scorable:
            logger.info("Computing BERTScore for %d responses", len(scorable))
            hypotheses, references, owners = [], [], []
            for record in scorable:
                for reference in record["references"]:
                    hypotheses.append(record["answer"])
                    references.append(reference)
                    owners.append(record)

            try:
                bert = self.bert_scorer.score(hypotheses, references)
                for record, scores in zip(owners, bert):
                    for key, value in scores.items():
                        record[key] = max(record.get(key, float("-inf")), value)
            except Exception as exc:  # noqa: BLE001
                logger.error("BERTScore failed, continuing without it: %s", exc)

        mode = "a" if append else "w"
        with output_path.open(mode, encoding="utf-8") as fh:
            if not append:
                fh.write(json.dumps(metadata.to_dict(), ensure_ascii=False) + "\n")
            for record in records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(
            "Wrote %d records for %s to %s in %.1fs",
            len(records), cfg.condition_id, output_path, elapsed,
        )
        return output_path


def run_conditions(
    configs: Sequence[PipelineConfig],
    benchmark: Benchmark,
    output_path: str | Path,
    metric_config: MetricConfig | None = None,
) -> Path:
    """Run several conditions into a single log file.

    Running the RAG and no-RAG conditions into one file is what makes the
    controlled comparison trivially correct: the two conditions are aggregated
    from the same file with the same code, so any difference between them
    cannot be an artefact of different scoring pipelines.
    """
    output_path = Path(output_path)
    metric_config = metric_config or MetricConfig()
    shared_scorer = BERTScorer(metric_config)

    for i, cfg in enumerate(configs):
        logger.info("Condition %d/%d: %s", i + 1, len(configs), cfg.condition_id)
        pipeline = RAGPipeline(cfg)
        runner = EvaluationRunner(pipeline, metric_config, shared_scorer)
        runner.run(benchmark, output_path, append=i > 0)

    return output_path
