"""Command-line entry points.

Every experiment in the paper corresponds to one command here. The intent is
that a reader can reproduce a table by running the command named in its
caption, rather than reconstructing a procedure from prose.

    python -m kannada_rag.cli build              # ingest + index
    python -m kannada_rag.cli ask "ಪ್ರಶ್ನೆ"        # single query
    python -m kannada_rag.cli evaluate           # run the benchmark
    python -m kannada_rag.cli tables             # regenerate all tables
    python -m kannada_rag.cli ablate retrieval   # hyperparameter sweeps
    python -m kannada_rag.cli ocr                # OCR CER/WER
    python -m kannada_rag.cli serve              # FastAPI server
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from .config import REPO_ROOT, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("kannada_rag")


def _load_config(path: str | None, **overrides) -> PipelineConfig:
    cfg = PipelineConfig.load(path) if path else PipelineConfig()
    generation_overrides = {
        k: v for k, v in overrides.items()
        if k in {"provider", "model", "num_repeats"} and v is not None
    }
    if generation_overrides:
        cfg = cfg.with_(generation=replace(cfg.generation, **generation_overrides))
    if overrides.get("no_retrieval"):
        cfg = cfg.with_(use_retrieval=False)
    return cfg


# --------------------------------------------------------------------- commands


def cmd_build(args) -> int:
    from .pipeline import build_index

    cfg = _load_config(args.config)
    index = build_index(cfg, force=args.force)
    print(f"Index ready: {len(index)} chunks in {cfg.index_dir}")
    print(f"Fingerprint: {cfg.index_fingerprint}")
    return 0


def cmd_ask(args) -> int:
    from .pipeline import RAGPipeline

    cfg = _load_config(args.config, provider=args.provider, model=args.model,
                       no_retrieval=args.no_retrieval)
    pipeline = RAGPipeline(cfg)
    answer = pipeline.answer(args.question)

    print(f"\nStatus: {answer.status}")
    print(f"Answer: {answer.answer}\n")
    if answer.sources:
        print("Sources:")
        for source in answer.sources:
            print(f"  - {source['citation']}  (similarity {source['score']:.3f})")
    print(f"\nRetrieval {answer.retrieval_latency_s:.3f}s, "
          f"generation {answer.generation_latency_s:.3f}s")
    return 0


def cmd_evaluate(args) -> int:
    from .eval.benchmark import load_benchmark
    from .eval.runner import run_conditions

    benchmark = load_benchmark(args.benchmark)
    problems = benchmark.validate()
    if problems:
        logger.warning("Benchmark validation notes:")
        for problem in problems:
            logger.warning("  %s", problem)
        if args.strict:
            logger.error("Refusing to run with --strict. Fix the benchmark first.")
            return 1

    base = _load_config(args.config, num_repeats=args.repeats)
    configs = []
    for model in args.models:
        configs.append(
            base.with_(generation=replace(base.generation, model=model, provider=args.provider))
        )
        if args.with_control:
            configs.append(
                base.with_(
                    generation=replace(base.generation, model=model, provider=args.provider),
                    use_retrieval=False,
                )
            )

    output = Path(args.output)
    run_conditions(configs, benchmark, output)
    print(f"Wrote raw records to {output}")
    print("Now run:  python -m kannada_rag.cli tables")
    return 0


def cmd_tables(args) -> int:
    from .eval.tables import build_all_tables

    try:
        tables = build_all_tables(args.log, args.output, strict=not args.allow_inconsistent)
    except ValueError as exc:
        logger.error("%s", exc)
        logger.error(
            "Tables were NOT written. An internally inconsistent results set must "
            "not reach the manuscript. Re-run the evaluation rather than editing "
            "the numbers."
        )
        return 1

    print(f"Wrote {len(tables)} tables to {args.output}")
    for name in tables:
        print(f"  {name}.tex / {name}.csv")
    return 0


def cmd_ablate(args) -> int:
    from .eval.ablations import (
        calibrate_threshold, sweep_chunking, sweep_embedding_models, sweep_retrieval,
    )
    from .eval.benchmark import load_benchmark

    cfg = _load_config(args.config)
    benchmark = load_benchmark(args.benchmark)
    output = Path(args.output)

    if args.what == "retrieval":
        table = sweep_retrieval(cfg, benchmark, output_dir=output)
    elif args.what == "chunking":
        table = sweep_chunking(cfg, benchmark, output_dir=output)
    elif args.what == "encoder":
        table = sweep_embedding_models(cfg, benchmark, output_dir=output)
    elif args.what == "threshold":
        payload = calibrate_threshold(cfg, benchmark, output_dir=output)
        print(json.dumps(payload["selected_threshold"], indent=2))
        return 0
    else:
        logger.error("Unknown ablation %r", args.what)
        return 1

    print(table.to_markdown())
    return 0


def cmd_ocr(args) -> int:
    from .eval.ocr_eval import evaluate_ocr, required_sample_size

    if args.sample_size:
        n = required_sample_size(
            expected_cer=args.expected_cer,
            half_width=args.half_width,
            population=args.population,
        )
        print(
            f"Transcribe {n} pages to estimate CER to within "
            f"+/-{args.half_width} at 95% confidence."
        )
        return 0

    result = evaluate_ocr(args.transcriptions, args.ingest_report)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "per_page"}, indent=2))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run("kannada_rag.api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


# ----------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kannada_rag",
        description="Kannada heritage RAG: build, query, evaluate, reproduce.",
    )
    parser.add_argument("--config", help="Path to a saved PipelineConfig JSON file.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="Ingest the corpus and build the vector index.")
    p.add_argument("--force", action="store_true", help="Rebuild even if a valid index exists.")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("ask", help="Answer a single Kannada question.")
    p.add_argument("question")
    p.add_argument("--provider", choices=["ollama", "groq", "echo"])
    p.add_argument("--model")
    p.add_argument("--no-retrieval", action="store_true",
                   help="Ungrounded control condition.")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("evaluate", help="Run the benchmark and write the raw record log.")
    p.add_argument("--benchmark", default=str(REPO_ROOT / "benchmark" / "questions.jsonl"))
    p.add_argument("--output", default=str(REPO_ROOT / "results" / "records.jsonl"))
    p.add_argument("--models", nargs="+", default=["gemma3:4b"])
    p.add_argument("--provider", default="ollama", choices=["ollama", "groq", "echo"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--with-control", action="store_true", default=True,
                   help="Also run each model without retrieval (the controlled ablation).")
    p.add_argument("--strict", action="store_true",
                   help="Refuse to run if the benchmark fails validation.")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("tables", help="Regenerate every table from the raw record log.")
    p.add_argument("--log", default=str(REPO_ROOT / "results" / "records.jsonl"))
    p.add_argument("--output", default=str(REPO_ROOT / "results" / "tables"))
    p.add_argument("--allow-inconsistent", action="store_true",
                   help="Write tables even if the consistency check fails. Do not use "
                        "for anything that goes into the paper.")
    p.set_defaults(func=cmd_tables)

    p = sub.add_parser("ablate", help="Run a design-choice sweep.")
    p.add_argument("what", choices=["retrieval", "chunking", "encoder", "threshold"])
    p.add_argument("--benchmark", default=str(REPO_ROOT / "benchmark" / "questions.jsonl"))
    p.add_argument("--output", default=str(REPO_ROOT / "results" / "ablations"))
    p.set_defaults(func=cmd_ablate)

    p = sub.add_parser("ocr", help="Measure OCR accuracy against manual transcriptions.")
    p.add_argument("--transcriptions", default=str(REPO_ROOT / "benchmark" / "ocr_transcriptions.jsonl"))
    p.add_argument("--ingest-report", default=str(REPO_ROOT / "vector_store" / "ingest_report.json"))
    p.add_argument("--output", default=str(REPO_ROOT / "results" / "ocr_eval.json"))
    p.add_argument("--sample-size", action="store_true",
                   help="Report how many pages need transcribing, then exit.")
    p.add_argument("--expected-cer", type=float, default=0.12)
    p.add_argument("--half-width", type=float, default=0.03)
    p.add_argument("--population", type=int, default=62)
    p.set_defaults(func=cmd_ocr)

    p = sub.add_parser("serve", help="Run the FastAPI server.")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
