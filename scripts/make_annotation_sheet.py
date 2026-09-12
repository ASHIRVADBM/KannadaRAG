#!/usr/bin/env python3
"""Build a CSV annotation sheet for human faithfulness judgements.

Splits each generated answer into claims and emits one row per claim, together
with the retrieved passages it should be checked against. Annotators fill the
`verdict` column; `scripts/score_annotations.py` reads it back.

    python scripts/make_annotation_sheet.py --log results/records.jsonl \
        --out results/annotation_sheet.csv --sample 60
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kannada_rag.eval.faithfulness import split_sentences  # noqa: E402
from kannada_rag.eval.tables import load_log  # noqa: E402

HEADER = [
    "question_id", "condition", "model", "claim_index", "question",
    "claim", "retrieved_context", "citations",
    "verdict",   # supported | contradicted | not_in_context | unverifiable
    "annotator", "notes",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", default="results/records.jsonl")
    parser.add_argument("--out", default="results/annotation_sheet.csv")
    parser.add_argument("--sample", type=int, default=0,
                        help="Sample this many answers (0 = all).")
    parser.add_argument("--seed", type=int, default=20240101)
    args = parser.parse_args()

    log = load_log(args.log)

    # One record per (question, condition): repeats add annotation cost without
    # adding evidence about grounding.
    seen: set[tuple[str, str]] = set()
    records = []
    for record in log.records:
        key = (record["question_id"], record["condition"])
        if key in seen or not record.get("answer"):
            continue
        seen.add(key)
        records.append(record)

    if args.sample and args.sample < len(records):
        random.Random(args.seed).shuffle(records)
        records = records[: args.sample]

    rows = []
    for record in records:
        passages = record.get("retrieved", []) or []
        context = "\n\n".join(
            f"[{i+1}] ({p.get('source_file','?')} p.{p.get('page','?')}) {p.get('text','')}"
            for i, p in enumerate(passages)
        )
        citations = "; ".join(
            f"{p.get('source_file','?')} p.{p.get('page','?')}" for p in passages
        )
        for i, claim in enumerate(split_sentences(record["answer"])):
            rows.append([
                record["question_id"], record["condition"], record.get("model", ""),
                i, record.get("question", ""), claim,
                context if i == 0 else "(see first row for this answer)",
                citations, "", "", "",
            ])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} claims from {len(records)} answers to {out}")
    print(
        "\nAnnotator instructions:\n"
        "  supported      - the claim follows from the retrieved context\n"
        "  contradicted   - the context says otherwise\n"
        "  not_in_context - the claim may be true but is absent from the context\n"
        "  unverifiable   - too vague to judge\n\n"
        "Have at least two annotators cover an overlapping subset so that "
        "Krippendorff's alpha can be reported."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
