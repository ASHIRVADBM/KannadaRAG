"""Evaluation: metrics, benchmark, runner, statistics and table generation.

The design rule of this subpackage: *one raw log, many derived tables*. Nothing
publishes a number that was not computed from
``results/records.jsonl`` by :mod:`kannada_rag.eval.tables`.
"""

from .benchmark import Benchmark, Question, load_benchmark
from .metrics import MetricConfig, chrf, meteor, rouge_l, rouge_n, score_pair
from .runner import EvaluationRunner, run_conditions
from .tables import build_all_tables, check_consistency, load_log

__all__ = [
    "Benchmark",
    "Question",
    "load_benchmark",
    "MetricConfig",
    "rouge_n",
    "rouge_l",
    "meteor",
    "chrf",
    "score_pair",
    "EvaluationRunner",
    "run_conditions",
    "build_all_tables",
    "check_consistency",
    "load_log",
]
