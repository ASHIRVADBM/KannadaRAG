# Reproducibility checklist

What must be recorded for a published number to be reproducible, and where this
repository records it.

## Recorded automatically in every run

| Item | Where |
|---|---|
| Full pipeline configuration | `_meta.config` in `results/records.jsonl` |
| Git commit | `_meta.git_commit` |
| Python version and platform | `_meta.python_version`, `_meta.platform` |
| Metric configuration | `_meta.metric_config` |
| BERTScore checkpoint, layer, IDF setting | `_meta.bertscore_provenance` |
| Benchmark provenance and independence fractions | `_meta.benchmark_provenance` |
| Index fingerprint | `vector_store/manifest.json` |
| Per-question retrieved passages and scores | every record's `retrieved` field |
| Reference version used for scoring | every record's `reference_version` |

## Must be recorded manually in the paper

- **Hardware.** CPU, GPU model, VRAM, RAM. Local latency is meaningless without it.
- **Ollama version and model digests.** `ollama list --format json` gives digests.
  A model tag such as `gemma3:4b` is not a version; the digest is.
- **Tesseract version and traineddata source.** `tesseract --version`. Kannada
  recognition accuracy differs substantially between traineddata releases.
- **Wall-clock conditions for any hosted-API measurement.** Date, time, region.
  Hosted latency is not a property of the model.

## Non-determinism

Three sources remain even with a fixed seed:

1. **LLM sampling.** `seed` is honoured by Ollama and is best-effort at best on
   hosted endpoints. Mitigated by `num_repeats` and by reporting variance.
2. **Floating-point non-associativity on GPU.** Embeddings can differ in the last
   few bits between runs and, very rarely, reorder near-tied retrieval results.
3. **OCR.** Deterministic for a fixed Tesseract version and traineddata;
   different across versions.

The correct response is to report intervals, not to claim exact reproducibility.

## What a reader should be able to do

```bash
git clone … && cd KannadaRAG
pip install -e ".[all]"
make reproduce
```

and obtain numbers that agree with the published ones within the reported
confidence intervals. If they do not, the discrepancy is a finding and the
configuration recorded in `_meta` is where to start looking.
