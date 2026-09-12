#!/usr/bin/env python3
"""How many benchmark questions are needed to detect a given difference?

Run this *before* claiming that one model outperforms another. If the benchmark
is too small to detect the difference being claimed, the honest report is the
interval, not the ranking.

    python scripts/power_analysis.py --effect 0.05 --power 0.8
    python scripts/power_analysis.py --from-log results/records.jsonl --effect 0.05
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def paired_n(effect: float, sd_diff: float, power: float, alpha: float) -> int:
    """Sample size for a paired two-sided test (normal approximation)."""
    from scipy import stats

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return math.ceil(((z_alpha + z_beta) * sd_diff / effect) ** 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--effect", type=float, default=0.05,
                        help="Smallest difference worth detecting.")
    parser.add_argument("--sd", type=float, default=0.12,
                        help="SD of the paired differences. Override with --from-log.")
    parser.add_argument("--power", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--comparisons", type=int, default=1,
                        help="Number of pairwise comparisons; alpha is Bonferroni-split.")
    parser.add_argument("--from-log", help="Estimate --sd from a real results log.")
    parser.add_argument("--metric", default="bert_f1")
    args = parser.parse_args()

    sd = args.sd
    if args.from_log:
        import numpy as np

        from kannada_rag.eval.tables import load_log

        log = load_log(args.from_log)
        models = log.models
        if len(models) < 2:
            print("Need at least two models in the log to estimate a paired SD.")
            return 1

        a = log.question_means(args.metric, model=models[0], use_retrieval=True, answerable=True)
        b = log.question_means(args.metric, model=models[1], use_retrieval=True, answerable=True)
        shared = sorted(set(a) & set(b))
        if len(shared) < 3:
            print("Too few shared questions to estimate an SD.")
            return 1
        differences = np.array([a[q] - b[q] for q in shared])
        sd = float(differences.std(ddof=1))
        print(f"Estimated SD of paired differences from {len(shared)} questions "
              f"({models[0]} vs {models[1]}, {args.metric}): {sd:.4f}\n")

    alpha = args.alpha / max(args.comparisons, 1)
    n = paired_n(args.effect, sd, args.power, alpha)

    print(f"To detect a difference of {args.effect:.3f} in {args.metric}")
    print(f"  paired SD              {sd:.4f}")
    print(f"  power                  {args.power:.0%}")
    print(f"  alpha                  {alpha:.4f}"
          + (f"  (Bonferroni over {args.comparisons} comparisons)" if args.comparisons > 1 else ""))
    print(f"\n  required questions:    {n}")

    if n > 25:
        print(f"\n  A 25-question benchmark is underpowered for this effect by a factor "
              f"of roughly {n / 25:.1f}. Either enlarge the benchmark or report the "
              f"interval rather than the ranking.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
