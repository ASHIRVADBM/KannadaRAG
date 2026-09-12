"""Derive every published table from the raw record log.

One input, many tables. The per-difficulty tables, the overall table, the
retrieval table, the latency table and the RAG/no-RAG comparison are all
aggregations of the *same* JSONL file, computed by the same code path. An
overall mean therefore cannot disagree with the per-difficulty means it
summarises, because it is computed from the identical records.

:func:`check_consistency` makes that guarantee explicit and testable. It
recomputes each overall figure as the question-count-weighted mean of the
per-difficulty figures and fails if they differ by more than floating-point
tolerance. Running it is a required step in ``make tables``; an internally
inconsistent results set cannot reach the manuscript without the build
failing first.

Output is emitted both as LaTeX (for the paper) and as CSV (for inspection),
from the same intermediate structure, so the two can never drift apart either.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .stats import describe, pairwise_model_comparison

logger = logging.getLogger(__name__)

__all__ = [
    "ResultLog",
    "Table",
    "load_log",
    "per_difficulty_table",
    "overall_table",
    "retrieval_table",
    "latency_table",
    "rag_ablation_table",
    "check_consistency",
    "build_all_tables",
]

DIFFICULTY_ORDER = ("easy", "medium", "hard", "unanswerable")

QUALITY_METRICS = (
    ("rouge1_f1", "ROUGE-1"),
    ("rouge2_f1", "ROUGE-2"),
    ("rougeL_f1", "ROUGE-L"),
    ("meteor", "METEOR"),
    ("chrf", "chrF"),
    ("bert_precision", "BERTScore P"),
    ("bert_recall", "BERTScore R"),
    ("bert_f1", "BERTScore F1"),
)

RETRIEVAL_METRICS = (
    ("recall@1", "R@1"),
    ("recall@3", "R@3"),
    ("recall@5", "R@5"),
    ("precision@5", "P@5"),
    ("mrr@10", "MRR@10"),
    ("ndcg@10", "nDCG@10"),
    ("evidence_hit", "Evidence hit"),
)


# --------------------------------------------------------------------- loading


@dataclass
class ResultLog:
    """Parsed record log: metadata plus per-question records."""

    meta: dict = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def models(self) -> list[str]:
        return sorted({r["model"] for r in self.records})

    @property
    def conditions(self) -> list[str]:
        return sorted({r["condition"] for r in self.records})

    def filter(self, **criteria) -> list[dict]:
        def matches(record: dict) -> bool:
            return all(record.get(k) == v for k, v in criteria.items())

        return [r for r in self.records if matches(r)]

    def values(self, metric: str, **criteria) -> list[float]:
        """Metric values, with missing and NaN entries dropped."""
        out = []
        for record in self.filter(**criteria):
            value = record.get(metric)
            if value is None:
                continue
            value = float(value)
            if math.isnan(value):
                continue
            out.append(value)
        return out

    def question_means(self, metric: str, **criteria) -> dict[str, float]:
        """Mean over repeats for each question, keyed by question_id.

        Averaging repeats *before* comparing models is what makes the paired
        tests legitimate: the unit of analysis is the question, not the
        individual generation.
        """
        buckets: dict[str, list[float]] = defaultdict(list)
        for record in self.filter(**criteria):
            value = record.get(metric)
            if value is None:
                continue
            value = float(value)
            if math.isnan(value):
                continue
            buckets[record["question_id"]].append(value)
        return {qid: sum(v) / len(v) for qid, v in buckets.items() if v}


def load_log(path: str | Path) -> ResultLog:
    path = Path(path)
    log = ResultLog()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "_meta" in payload:
                log.meta = payload["_meta"]
                continue
            log.records.append(payload)
    logger.info("Loaded %d records from %s", len(log.records), path)
    return log


# ----------------------------------------------------------------------- table


@dataclass
class Table:
    """A rendered table, emitted identically to LaTeX and CSV."""

    caption: str
    label: str
    columns: list[str]
    rows: list[list[Any]]
    notes: str = ""

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(self.columns)
            writer.writerows(self.rows)
        return path

    def to_latex(self) -> str:
        alignment = "l" + "c" * (len(self.columns) - 1)
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            f"\\caption{{{self.caption}}}",
            f"\\label{{{self.label}}}",
            f"\\begin{{tabular}}{{{alignment}}}",
            r"\hline",
            " & ".join(str(c) for c in self.columns) + r" \\",
            r"\hline",
        ]
        for row in self.rows:
            lines.append(" & ".join(_fmt(v) for v in row) + r" \\")
        lines += [r"\hline", r"\end{tabular}"]
        if self.notes:
            lines.append(f"\\\\[2pt]\\footnotesize {self.notes}")
        lines.append(r"\end{table}")
        return "\n".join(lines)

    def to_markdown(self) -> str:
        header = "| " + " | ".join(str(c) for c in self.columns) + " |"
        divider = "| " + " | ".join("---" for _ in self.columns) + " |"
        body = [
            "| " + " | ".join(_fmt(v) for v in row) + " |" for row in self.rows
        ]
        out = [f"**{self.caption}**", "", header, divider, *body]
        if self.notes:
            out += ["", f"_{self.notes}_"]
        return "\n".join(out)


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if math.isnan(value):
            return "--"
        return f"{value:.3f}"
    return str(value)


def _cell(values: Sequence[float], with_ci: bool) -> str:
    """Format a metric cell, optionally with its bootstrap interval."""
    if not values:
        return "--"
    stats = describe(values)
    if not with_ci:
        return f"{stats['mean']:.3f}"
    return f"{stats['mean']:.3f} [{stats['ci_low']:.3f}, {stats['ci_high']:.3f}]"


# ---------------------------------------------------------------------- tables


def per_difficulty_table(
    log: ResultLog,
    difficulty: str,
    metrics: Sequence[tuple[str, str]] = QUALITY_METRICS,
    with_ci: bool = True,
    use_retrieval: bool = True,
) -> Table:
    """Response quality for one difficulty tier, one row per model."""
    models = log.models
    columns = ["Model", "n"] + [label for _, label in metrics]
    rows: list[list[Any]] = []

    for model in models:
        criteria = {"model": model, "difficulty": difficulty, "use_retrieval": use_retrieval}
        records = log.filter(**criteria)
        if not records:
            continue
        n_questions = len({r["question_id"] for r in records})
        row: list[Any] = [model, n_questions]
        for key, _ in metrics:
            row.append(_cell(list(log.question_means(key, **criteria).values()), with_ci))
        rows.append(row)

    return Table(
        caption=(
            f"{difficulty.capitalize()}-tier response quality, averaged over "
            f"{log.meta.get('config', {}).get('generation', {}).get('num_repeats', 1)} "
            "repeats per question. Brackets give 95\\% bootstrap confidence intervals."
        ),
        label=f"tab:quality-{difficulty}",
        columns=columns,
        rows=rows,
        notes="All values derived from the same per-question log; see results/records.jsonl.",
    )


def overall_table(
    log: ResultLog,
    metrics: Sequence[tuple[str, str]] = QUALITY_METRICS,
    with_ci: bool = True,
    use_retrieval: bool = True,
) -> Table:
    """Response quality over the whole answerable benchmark.

    This is an aggregation of exactly the records that feed the per-difficulty
    tables, restricted to answerable questions. :func:`check_consistency`
    verifies that it equals their count-weighted mean.
    """
    models = log.models
    columns = ["Model", "n"] + [label for _, label in metrics]
    rows: list[list[Any]] = []

    for model in models:
        criteria = {"model": model, "use_retrieval": use_retrieval, "answerable": True}
        records = log.filter(**criteria)
        if not records:
            continue
        n_questions = len({r["question_id"] for r in records})
        row: list[Any] = [model, n_questions]
        for key, _ in metrics:
            row.append(_cell(list(log.question_means(key, **criteria).values()), with_ci))
        rows.append(row)

    return Table(
        caption=(
            "Overall response quality across the full answerable benchmark. "
            "Each value is the mean over per-question means; brackets give 95\\% "
            "bootstrap confidence intervals."
        ),
        label="tab:quality-overall",
        columns=columns,
        rows=rows,
        notes=(
            "Aggregated from the same records as the per-tier tables. "
            "Verified by tests/test_table_consistency.py."
        ),
    )


def retrieval_table(log: ResultLog, with_ci: bool = True) -> Table:
    """Retrieval quality against annotated gold evidence.

    Independent of the generator: retrieval is identical across models for a
    given index, so this table has one row per difficulty tier rather than one
    per model.
    """
    columns = ["Tier", "n"] + [label for _, label in RETRIEVAL_METRICS]
    rows: list[list[Any]] = []

    for difficulty in DIFFICULTY_ORDER:
        criteria = {"difficulty": difficulty, "use_retrieval": True}
        records = log.filter(**criteria)
        if not records:
            continue
        n_questions = len({r["question_id"] for r in records})
        row: list[Any] = [difficulty, n_questions]
        for key, _ in RETRIEVAL_METRICS:
            row.append(_cell(list(log.question_means(key, **criteria).values()), with_ci))
        rows.append(row)

    # All-tier row.
    all_records = log.filter(use_retrieval=True, answerable=True)
    if all_records:
        row = ["all (answerable)", len({r["question_id"] for r in all_records})]
        for key, _ in RETRIEVAL_METRICS:
            row.append(
                _cell(
                    list(
                        log.question_means(key, use_retrieval=True, answerable=True).values()
                    ),
                    with_ci,
                )
            )
        rows.append(row)

    return Table(
        caption=(
            "Retrieval quality against manually annotated gold evidence passages, "
            "computed over the unfiltered ranked candidate list. "
            "Evidence hit is measured on the post-threshold selection actually "
            "shown to the generator."
        ),
        label="tab:retrieval",
        columns=columns,
        rows=rows,
    )


def latency_table(log: ResultLog) -> Table:
    """Latency, with retrieval and generation reported separately.

    Local and hosted models are placed in separate blocks and never averaged
    together: a hosted figure includes network round-trip and queueing time
    that has nothing to do with the model or the pipeline.
    """
    columns = [
        "Model", "Deployment", "n",
        "Retrieval (s)", "Generation (s)", "End-to-end (s)", "Answer length (tokens)",
    ]
    rows: list[list[Any]] = []

    for model in log.models:
        records = log.filter(model=model, use_retrieval=True)
        if not records:
            continue
        is_local = records[0].get("is_local")
        deployment = "local" if is_local else "hosted API"

        rows.append(
            [
                model,
                deployment,
                len(records),
                _cell(log.values("retrieval_latency_s", model=model, use_retrieval=True), True),
                _cell(log.values("generation_latency_s", model=model, use_retrieval=True), True),
                _cell(log.values("total_latency_s", model=model, use_retrieval=True), True),
                _cell(log.values("hyp_tokens", model=model, use_retrieval=True), True),
            ]
        )

    rows.sort(key=lambda r: (r[1] != "local", r[0]))

    return Table(
        caption=(
            "Computational cost. Retrieval and generation latency are reported "
            "separately; brackets give 95\\% bootstrap confidence intervals over "
            "all queries and repeats."
        ),
        label="tab:latency",
        columns=columns,
        rows=rows,
        notes=(
            "Local measurements were taken on the hardware described in the "
            "experimental setup. Hosted-API latency includes network round-trip "
            "and provider-side queueing and is not comparable to local latency."
        ),
    )


def rag_ablation_table(log: ResultLog, metric: str = "bert_f1") -> Table:
    """The controlled comparison: same model, same prompt, with and without retrieval.

    This is the table that supports any claim that retrieval improves factual
    accuracy. Both conditions use identical decoding settings and an identical
    prompt apart from the context block, so the difference is attributable to
    retrieval rather than to model choice or sampling.
    """
    columns = [
        "Model", "With retrieval", "Without retrieval", "Difference [95% CI]",
        "Wilcoxon p (Holm)", "Cliff's delta",
    ]
    rows: list[list[Any]] = []

    from .stats import paired_comparison, holm_bonferroni

    comparisons = []
    for model in log.models:
        with_rag = log.question_means(metric, model=model, use_retrieval=True, answerable=True)
        without_rag = log.question_means(metric, model=model, use_retrieval=False, answerable=True)
        shared = sorted(set(with_rag) & set(without_rag))
        if len(shared) < 2:
            continue
        comparisons.append(
            paired_comparison(
                [with_rag[q] for q in shared],
                [without_rag[q] for q in shared],
                name_a=f"{model}+RAG",
                name_b=f"{model} (no retrieval)",
                metric=metric,
            )
        )

    holm_bonferroni(comparisons)

    for comparison in comparisons:
        rows.append(
            [
                comparison.name_a.replace("+RAG", ""),
                f"{comparison.mean_a:.3f}",
                f"{comparison.mean_b:.3f}",
                f"{comparison.mean_difference:+.3f} "
                f"[{comparison.difference_ci[0]:+.3f}, {comparison.difference_ci[1]:+.3f}]",
                f"{comparison.p_adjusted:.4f}" if comparison.p_adjusted is not None else "--",
                f"{comparison.cliffs_delta:+.2f} ({comparison.effect_label})",
            ]
        )

    return Table(
        caption=(
            f"Controlled retrieval ablation on {metric}. Each model answers the same "
            "questions twice under identical decoding settings, once with retrieved "
            "context and once without. p-values are Holm-corrected across models."
        ),
        label="tab:rag-ablation",
        columns=columns,
        rows=rows,
    )


def significance_table(log: ResultLog, metric: str = "bert_f1") -> Table:
    """All pairwise model comparisons, Holm-corrected."""
    per_model: dict[str, list[float]] = {}
    question_ids: set[str] | None = None

    for model in log.models:
        means = log.question_means(metric, model=model, use_retrieval=True, answerable=True)
        if not means:
            continue
        per_model[model] = means
        question_ids = set(means) if question_ids is None else question_ids & set(means)

    if not per_model or not question_ids:
        return Table("No comparable data", "tab:significance", ["Note"], [["insufficient data"]])

    shared = sorted(question_ids)
    aligned = {m: [v[q] for q in shared] for m, v in per_model.items()}
    comparisons = pairwise_model_comparison(aligned, metric=metric)

    return Table(
        caption=(
            f"Pairwise model comparisons on {metric} over {len(shared)} shared questions. "
            "Paired Wilcoxon signed-rank tests, Holm-corrected for multiple comparisons."
        ),
        label="tab:significance",
        columns=["Model A", "Model B", "Difference [95% CI]", "p (Holm)", "Cliff's delta", "Significant"],
        rows=[
            [
                c.name_a,
                c.name_b,
                f"{c.mean_difference:+.3f} [{c.difference_ci[0]:+.3f}, {c.difference_ci[1]:+.3f}]",
                f"{c.p_adjusted:.4f}" if c.p_adjusted is not None else "--",
                f"{c.cliffs_delta:+.2f} ({c.effect_label})",
                "yes" if c.significant else "no",
            ]
            for c in comparisons
        ],
        notes=(
            "At this sample size most pairwise differences are not separable from "
            "sampling noise; that is a property of the benchmark size, not evidence "
            "of equivalence."
        ),
    )


# ----------------------------------------------------------------- consistency


def check_consistency(log: ResultLog, tolerance: float = 1e-9) -> list[str]:
    """Verify that overall figures equal the weighted per-tier figures.

    Returns a list of violations; an empty list means the results set is
    internally consistent. This is the check whose absence produced an overall
    BERT-F1 that no combination of the per-tier values could yield.
    """
    problems: list[str] = []

    for model in log.models:
        for metric, label in QUALITY_METRICS:
            overall = log.question_means(
                metric, model=model, use_retrieval=True, answerable=True
            )
            if not overall:
                continue
            overall_mean = sum(overall.values()) / len(overall)

            weighted_total = 0.0
            weight_total = 0
            for difficulty in DIFFICULTY_ORDER:
                tier = log.question_means(
                    metric, model=model, difficulty=difficulty, use_retrieval=True
                )
                tier = {
                    qid: v
                    for qid, v in tier.items()
                    if qid in overall
                }
                if not tier:
                    continue
                weighted_total += sum(tier.values())
                weight_total += len(tier)

            if weight_total == 0:
                continue
            reconstructed = weighted_total / weight_total

            if abs(reconstructed - overall_mean) > tolerance:
                problems.append(
                    f"{model} / {label}: overall mean {overall_mean:.6f} != "
                    f"weighted tier mean {reconstructed:.6f} "
                    f"(difference {overall_mean - reconstructed:+.2e})"
                )

    # Every record must name the reference version it was scored against.
    versions = {r.get("reference_version") for r in log.records if r.get("answerable")}
    if len(versions) > 1:
        problems.append(f"records scored against mixed reference versions: {sorted(versions)}")
    if None in versions:
        problems.append("some records carry no reference_version")

    return problems


# ------------------------------------------------------------------ build all


def build_all_tables(
    log_path: str | Path,
    output_dir: str | Path,
    strict: bool = True,
) -> dict[str, Table]:
    """Regenerate every manuscript table from one log file.

    With ``strict=True`` (the default, and what ``make tables`` uses) an
    internal inconsistency raises rather than writing output. A results set
    that cannot pass its own consistency check must not reach the paper.
    """
    log = load_log(log_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    problems = check_consistency(log)
    if problems:
        message = "Internal inconsistency in results:\n  " + "\n  ".join(problems)
        if strict:
            raise ValueError(message)
        logger.warning(message)

    tables: dict[str, Table] = {}
    for difficulty in ("easy", "medium", "hard"):
        if log.filter(difficulty=difficulty):
            tables[f"quality_{difficulty}"] = per_difficulty_table(log, difficulty)

    tables["quality_overall"] = overall_table(log)
    if any("recall@5" in r for r in log.records):
        tables["retrieval"] = retrieval_table(log)
    tables["latency"] = latency_table(log)
    if log.filter(use_retrieval=False):
        tables["rag_ablation"] = rag_ablation_table(log)
    tables["significance"] = significance_table(log)

    latex_parts = []
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv")
        (output_dir / f"{name}.tex").write_text(table.to_latex(), encoding="utf-8")
        latex_parts.append(table.to_latex())

    (output_dir / "all_tables.tex").write_text("\n\n".join(latex_parts), encoding="utf-8")
    (output_dir / "all_tables.md").write_text(
        "\n\n".join(t.to_markdown() for t in tables.values()), encoding="utf-8"
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(
            {
                "source_log": str(log_path),
                "n_records": len(log),
                "consistency_problems": problems,
                "run_metadata": log.meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info("Wrote %d tables to %s", len(tables), output_dir)
    return tables
