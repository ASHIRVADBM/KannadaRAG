#!/usr/bin/env python3
"""Report the realised chunk-length distribution.

The paper should state what chunk lengths actually occurred, not the nominal
configuration value: the separator hierarchy rarely saturates the nominal size,
so "500 characters" is an upper bound rather than a description.

    python scripts/report_chunk_stats.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kannada_rag.chunking import chunk_length_stats, chunk_pages  # noqa: E402
from kannada_rag.config import PipelineConfig  # noqa: E402
from kannada_rag.ingest import ingest_corpus  # noqa: E402


def main() -> int:
    cfg = PipelineConfig()
    report = ingest_corpus(cfg.data_dir, cfg.ingest)
    chunks = chunk_pages(report.pages, cfg.chunk)
    stats = chunk_length_stats(chunks)

    print(json.dumps({"ingest": report.summary(), "chunks": stats},
                     indent=2, ensure_ascii=False))

    chars = stats["characters"]
    tokens = stats["tokens"]
    print(
        f"\nFor the manuscript:\n"
        f"  Nominal chunk size {cfg.chunk.chunk_size} characters with "
        f"{cfg.chunk.chunk_overlap}-character overlap yields {stats['n_chunks']} chunks\n"
        f"  with a median length of {chars['median']} characters "
        f"(IQR {chars['p25']}-{chars['p75']}, range {chars['min']}-{chars['max']}),\n"
        f"  corresponding to a median of {tokens['median']} Kannada tokens "
        f"(IQR {tokens['p25']}-{tokens['p75']}).\n"
        f"  {stats['n_ocr_derived_chunks']} chunks derive from OCR-recovered pages."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
