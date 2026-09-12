"""The test that would have caught the published table inconsistency.

An overall metric value must equal the question-weighted mean of the
per-difficulty values computed from the same records. This file builds a
synthetic log, verifies that the real aggregation path satisfies that identity,
and then verifies that :func:`check_consistency` actually *detects* a violation
when one is introduced -- a consistency check that cannot fail is worthless.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kannada_rag.eval.retrieval_metrics import (
    mrr, ndcg_at_k, precision_at_k, recall_at_k,
)
from kannada_rag.eval.stats import holm_bonferroni, paired_comparison
from kannada_rag.eval.tables import check_consistency, load_log, overall_table


def _write_log(path: Path, records: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"_meta": {"config": {"generation": {"num_repeats": 1}}}}) + "\n")
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _record(qid, difficulty, model, bert_f1, **extra):
    return {
        "question_id": qid,
        "difficulty": difficulty,
        "model": model,
        "condition": f"{model}__rag",
        "use_retrieval": True,
        "answerable": True,
        "reference_version": "v1",
        "repeat": 0,
        "status": "ok",
        "answer": "…",
        "references": ["…"],
        "bert_f1": bert_f1,
        **extra,
    }


class TestConsistency:
    def test_overall_equals_weighted_tier_mean(self, tmp_path: Path):
        records = [
            _record("e1", "easy", "gemma3:4b", 0.85),
            _record("e2", "easy", "gemma3:4b", 0.87),
            _record("m1", "medium", "gemma3:4b", 0.80),
            _record("h1", "hard", "gemma3:4b", 0.70),
        ]
        log = load_log(_write_log(tmp_path / "records.jsonl", records))

        assert check_consistency(log) == []

        # And the derived overall value is the plain mean of the four, which is
        # what "aggregate of the tiers" has to mean.
        values = list(log.question_means("bert_f1", model="gemma3:4b",
                                         use_retrieval=True, answerable=True).values())
        assert sum(values) / len(values) == pytest.approx((0.85 + 0.87 + 0.80 + 0.70) / 4)

    def test_detects_an_orphaned_overall_record(self, tmp_path: Path):
        """A record present in the overall pool but in no difficulty tier.

        This reproduces the structural shape of the published error: an overall
        figure computed over a different set of records than the per-tier
        figures.
        """
        records = [
            _record("e1", "easy", "gemma3:4b", 0.85),
            _record("m1", "medium", "gemma3:4b", 0.86),
            _record("h1", "hard", "gemma3:4b", 0.85),
            # Same model, answerable, retrieval on, but a difficulty label that
            # no per-tier table covers -- and a wildly different score.
            _record("x1", "bonus", "gemma3:4b", 0.99),
        ]
        log = load_log(_write_log(tmp_path / "records.jsonl", records))

        problems = check_consistency(log)
        assert problems, "an overall/tier mismatch must be reported"
        assert "gemma3:4b" in problems[0]

    def test_detects_mixed_reference_versions(self, tmp_path: Path):
        records = [
            _record("e1", "easy", "gemma3:4b", 0.85),
            _record("e2", "easy", "gemma3:4b", 0.86),
        ]
        records[1]["reference_version"] = "v2"
        log = load_log(_write_log(tmp_path / "records.jsonl", records))

        problems = check_consistency(log)
        assert any("reference version" in p for p in problems)

    def test_overall_table_reports_the_right_question_count(self, tmp_path: Path):
        records = [
            _record(f"q{i}", "easy" if i < 8 else "medium", "gemma3:4b", 0.8)
            for i in range(17)
        ]
        log = load_log(_write_log(tmp_path / "records.jsonl", records))

        table = overall_table(log, with_ci=False)
        assert table.rows[0][1] == 17

    def test_repeats_are_averaged_before_aggregation(self, tmp_path: Path):
        """The unit of analysis is the question, not the generation."""
        records = [
            _record("e1", "easy", "gemma3:4b", 0.80, repeat=0),
            _record("e1", "easy", "gemma3:4b", 0.90, repeat=1),
            _record("e2", "easy", "gemma3:4b", 0.60, repeat=0),
            _record("e2", "easy", "gemma3:4b", 0.60, repeat=1),
        ]
        log = load_log(_write_log(tmp_path / "records.jsonl", records))

        means = log.question_means("bert_f1", model="gemma3:4b", use_retrieval=True)
        assert means == {"e1": pytest.approx(0.85), "e2": pytest.approx(0.60)}
        assert len(means) == 2, "two questions, not four generations"


class TestRetrievalMetrics:
    def test_recall_and_precision_at_k(self):
        retrieved = ["a", "b", "c", "d", "e"]
        gold = ["c", "z"]
        assert recall_at_k(retrieved, gold, 5) == pytest.approx(0.5)
        assert recall_at_k(retrieved, gold, 2) == 0.0
        assert precision_at_k(retrieved, gold, 5) == pytest.approx(0.2)

    def test_mrr_uses_first_relevant_rank(self):
        assert mrr(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)
        assert mrr(["c", "b", "a"], ["c"]) == pytest.approx(1.0)
        assert mrr(["a", "b"], ["z"]) == 0.0

    def test_ndcg_is_one_for_ideal_ranking(self):
        assert ndcg_at_k(["a", "b", "c"], ["a", "b"], 3) == pytest.approx(1.0)

    def test_ndcg_penalises_late_relevant_items(self):
        ideal = ndcg_at_k(["a", "x", "y"], ["a"], 3)
        late = ndcg_at_k(["x", "y", "a"], ["a"], 3)
        assert late < ideal


class TestStatistics:
    def test_paired_comparison_finds_a_real_difference(self):
        a = [0.9, 0.88, 0.91, 0.87, 0.92, 0.89, 0.90, 0.93]
        b = [0.7, 0.68, 0.71, 0.67, 0.72, 0.69, 0.70, 0.73]
        result = paired_comparison(a, b, "A", "B", "bert_f1")

        assert result.mean_difference == pytest.approx(0.2, abs=0.01)
        assert result.wilcoxon_p < 0.05
        assert result.cliffs_delta == pytest.approx(1.0)
        assert result.effect_label == "large"

    def test_identical_inputs_are_not_significant(self):
        values = [0.8, 0.7, 0.9, 0.6]
        result = paired_comparison(values, values)
        assert result.mean_difference == 0.0
        assert result.wilcoxon_p == 1.0

    def test_holm_is_monotonic_and_no_smaller_than_raw(self):
        comparisons = [
            paired_comparison([0.9] * 6, [0.1] * 6, f"a{i}", f"b{i}")
            for i in range(3)
        ]
        for c in comparisons:
            c.wilcoxon_p = 0.01 * (comparisons.index(c) + 1)

        holm_bonferroni(comparisons)
        adjusted = [c.p_adjusted for c in comparisons]

        assert all(adj >= raw for adj, raw in zip(adjusted, [0.01, 0.02, 0.03]))
        assert adjusted == sorted(adjusted)
