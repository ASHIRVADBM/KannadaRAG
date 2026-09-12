<div align="center">

# Kannada Heritage RAG

**ಕರ್ನಾಟಕ ಪಾರಂಪರಿಕ ಪ್ರಶ್ನೋತ್ತರ ವ್ಯವಸ್ಥೆ**

Retrieval-augmented question answering over Karnataka heritage documents — in Kannada,
grounded in cited source pages, and deployable entirely offline.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FAISS](https://img.shields.io/badge/retrieval-FAISS-0467DF)](https://faiss.ai/)
[![Ollama](https://img.shields.io/badge/inference-Ollama%20(local)-000000)](https://ollama.com/)
[![Tests](https://img.shields.io/badge/tests-35%20passing-2f6b45)](tests/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Quick start](#quick-start) · [How it works](#how-it-works) · [Reproducing the results](#reproducing-the-results) · [Evaluation design](#evaluation-design)

</div>

---

## What this is

General-purpose language models answer poorly in Kannada and fabricate freely on
Karnataka's regional history. Kannada makes up well under 1% of the web-scale
corpora these models are trained on, and much of the state's heritage record
exists only as scanned Kannada-language documents that never entered those
corpora at all.

This system answers Kannada questions about Karnataka heritage **only** from a
curated document corpus, and returns the document and page each answer came
from. When the corpus does not contain the answer, it says so rather than
generating a plausible one.

Three properties distinguish it from prompting a language model directly:

- **Grounded.** Every answer is generated from retrieved passages. No retrieval,
  no answer — the system abstains instead.
- **Traceable.** Every response carries its source document, page number,
  retrieval similarity score, and whether the passage was recovered by OCR.
- **Local.** The full pipeline runs on a single consumer GPU through Ollama. No
  cloud API, no data leaving the machine — which matters for deployment at
  heritage sites and institutions with unreliable connectivity.

---

## Quick start

### Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) with at least one model pulled
- Tesseract OCR with Kannada language data (only if your PDFs contain scanned pages)

<details>
<summary><b>Installing Tesseract with Kannada support</b></summary>

```bash
# Debian / Ubuntu
sudo apt install tesseract-ocr tesseract-ocr-kan

# macOS
brew install tesseract tesseract-lang

# Verify
tesseract --list-langs | grep kan
```

On Windows, install the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki),
selecting **Kannada** under additional language data, and add the install
directory to `PATH`.
</details>

### Install and run

```bash
git clone https://github.com/ASHIRVADBM/KannadaRAG.git
cd KannadaRAG

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[all]"

ollama pull gemma3:4b

# Put your Kannada heritage PDFs in data/, then:
make index
make serve
```

Open <http://127.0.0.1:8000> and ask a question in Kannada.

No API key is required. `.env` is only needed for the optional hosted-model
comparison; copy `.env.example` to `.env` if you want it.

### Command line

```bash
# One question
python -m kannada_rag.cli ask "ಹಂಪಿಯ ಐತಿಹಾಸಿಕ ಮಹತ್ವವೇನು?"

# The same question without retrieval — the controlled comparison
python -m kannada_rag.cli ask "ಹಂಪಿಯ ಐತಿಹಾಸಿಕ ಮಹತ್ವವೇನು?" --no-retrieval
```

---

## How it works

```
OFFLINE  (once, at index build)              ONLINE  (per query)
─────────────────────────────────            ──────────────────────────────────
  Kannada PDFs                                 Kannada question
       │                                             │
       ▼                                             ▼
  PyMuPDF native text                          Language guard
       │  < 50 chars?                          (≥50% Kannada script,
       ▼                                        else refuse and explain)
  Tesseract OCR @ 200 DPI                            │
  (kept only if longer)                              ▼
       │                                       Encode query
       ▼                                       (same encoder)
  Unicode NFC normalisation                          │
  danda + whitespace cleanup                         ▼
       │                                       FAISS cosine search, k_max
       ▼                                             │
  Recursive chunking                                 ▼
  500 chars / 50 overlap                       Selection rule
  (paragraph → line → danda → word)            sim ≥ 0.55  AND
       │                                       sim ≥ 0.85 × top
       ▼                                             │
  paraphrase-multilingual-mpnet-base-v2        ┌──── nothing passes?
       │                                       │          │
       ▼                                       │          ▼
  FAISS IndexFlatIP over                       │     ABSTAIN
  L2-normalised vectors  ──────────────────────┘     "not in the document"
  (inner product = cosine)                           │
       │                                             ▼
       ▼                                       Kannada grounded prompt
  chunks.jsonl (text + page provenance)              │
  manifest.json (config fingerprint)                 ▼
                                               Local LLM via Ollama
                                               temp 0.3, 150 tokens
                                                     │
                                                     ▼
                                               Answer + cited sources
```

### Design decisions worth knowing

**Cosine similarity, not L2 distance.** The index is `IndexFlatIP` over
L2-normalised vectors, so scores are cosine similarities in `[-1, 1]` where
higher is better, and the 0.55 threshold is a similarity **floor**. This is
stated explicitly because the direction of the comparison is easy to get
backwards.

**K is computed, not chosen by hand.** Two rules decide how many passages reach
the generator: an absolute floor (`sim ≥ 0.55`) that determines whether the
corpus contains anything relevant at all, and a relative floor
(`sim ≥ 0.85 × top_score`) that determines how many passages are comparably
relevant to the best one. A query with one dominant hit gets a narrow context;
a query with several close hits gets a wider one. Both thresholds are calibrated
on a development split, not on the evaluation questions:

```bash
python -m kannada_rag.cli ablate threshold
```

**Abstention is a feature.** When nothing clears the absolute floor, the system
returns `status: abstained` and a fixed Kannada message. Measuring this
behaviour on deliberately unanswerable questions is the cleanest automatic
evidence that the system is grounded rather than merely fluent.

**One config, one fingerprint.** Every parameter lives in
[`config.py`](src/kannada_rag/config.py). The ingestion, chunking and embedding
settings are hashed into a fingerprint stored with the index; an index built
under different settings is refused on load rather than silently reused.

---

## Reproducing the results

Every table in the accompanying paper is derived from a single raw log by a
single code path. There are no separate scripts producing separate tables.

```bash
make index                  # ingest + embed + index
make evaluate               # run the benchmark, all models, ± retrieval
make tables                 # derive every table from results/records.jsonl
make ablate                 # threshold, retrieval, chunking, encoder sweeps
make ocr                    # CER/WER against manual transcriptions
```

`make evaluate` writes `results/records.jsonl` — one record per
(question × condition × repeat), each containing the question id, reference
version, model output, retrieved passages with scores, every per-question metric
value, and the full configuration that produced it.

`make tables` then aggregates that file into `results/tables/`, as both `.tex`
and `.csv`. Before writing anything it runs a consistency check that recomputes
each overall figure as the question-weighted mean of the per-tier figures and
**fails the build** if they disagree. An internally inconsistent results set
cannot reach the manuscript.

```
results/
├── records.jsonl          ← the single source of truth
├── tables/
│   ├── quality_easy.tex|csv
│   ├── quality_medium.tex|csv
│   ├── quality_hard.tex|csv
│   ├── quality_overall.tex|csv      ← aggregate of the three above, by construction
│   ├── retrieval.tex|csv
│   ├── latency.tex|csv
│   ├── rag_ablation.tex|csv         ← same model, ± retrieval
│   ├── significance.tex|csv
│   └── provenance.json
├── ablations/
└── ocr_eval.json
```

---

## Evaluation design

### Response quality

Metrics are implemented to their published definitions, and the tests pin each
one to a hand-computable case.

| Metric | Note |
|---|---|
| ROUGE-1 / ROUGE-2 / ROUGE-L | F1 with clipped n-gram counts; ROUGE-L over the LCS |
| **METEOR** | The complete metric — unigram alignment, recall-weighted `F_mean = 10PR/(R+9P)`, **and** the fragmentation penalty `γ(chunks/matches)^β`. A "METEOR" without the penalty is a different metric and is not called METEOR here. |
| **chrF** | Character n-gram F-score. The metric best suited to an agglutinative script: inflected forms of one Kannada lemma share most character n-grams and no word unigrams. |
| BERTScore | `xlm-roberta-base`, layer 9, IDF off — all three pinned in config and recorded in every output file, because a BERTScore is not reproducible without them. |

The retrieval encoder (`paraphrase-multilingual-mpnet-base-v2`) and the
evaluation encoder (`xlm-roberta-base`) are kept strictly separate. They are
different models serving different purposes and are never interchanged.

### Retrieval quality

Measured directly against human-annotated evidence passages, over the
**unfiltered** ranked candidate list, so retrieval quality is separable from the
abstention threshold:

Recall@{1,3,5,10} · Precision@k · MRR@10 · nDCG@10 · evidence-hit rate

A high answer score is compatible with retrieval having failed entirely — a
fluent model can produce a plausible Kannada sentence from a weak context — so
answer quality is not evidence about retrieval.

### Grounding and hallucination

BERTScore cannot establish factual grounding; it measures similarity to a
reference, and a confident fabrication scores well. Three measurements are used
instead, in increasing order of evidential weight:

1. **Support score** (automatic) — per-sentence character-n-gram containment of
   the answer in the retrieved context. The *minimum* across sentences is
   reported alongside the mean, because one unsupported sentence appended to
   three grounded ones is the failure mode of interest and a mean hides it.
2. **Abstention behaviour** (automatic) — on the unanswerable tier. Abstention
   recall is reported with the false-abstention rate on answerable questions,
   since a system that declines everything scores perfectly on the first.
3. **Human claim judgements** — each claim marked supported / contradicted /
   not-in-context / unverifiable, with Krippendorff's α for inter-annotator
   agreement. Automatic proxies indicate; human judgements establish.

### Statistics

With a small question set, differences of a few points are inside sampling
noise. Every aggregate carries a 95% bootstrap percentile interval; every model
comparison is a **paired** Wilcoxon signed-rank test (with the paired t-test
reported alongside), Holm–Bonferroni corrected across comparisons, with Cliff's
delta as an effect size.

Reporting that most pairwise differences do not survive correction at this
sample size is itself a finding, and an honest one.

### The controlled retrieval comparison

Claims that retrieval helps are supported by the only comparison that isolates
it: **the same model, the same prompt, the same decoding settings, run twice** —
once with retrieved context and once without. Comparing a local grounded model
against a hosted ungrounded model under the provider's own default sampling
settings measures neither retrieval nor the model.

```bash
python -m kannada_rag.cli evaluate --models gemma3:4b --with-control
```

---

## Project layout

```
src/kannada_rag/
├── config.py          Every tunable parameter, defined exactly once
├── text.py            Kannada NFC normalisation, language guard, metric tokeniser
├── ingest.py          Hybrid PyMuPDF/Tesseract extraction with OCR provenance
├── chunking.py        Recursive chunking; provenance inherited by every chunk
├── index.py           FAISS cosine index storing chunk text, not just metadata
├── retriever.py       The absolute/relative selection rule and abstention
├── prompts.py         Grounded and ungrounded templates, kept symmetric
├── generation.py      Ollama / Groq / test providers under matched decoding
├── pipeline.py        End-to-end, both conditions
├── api.py             FastAPI service
├── cli.py             One command per experiment
└── eval/
    ├── benchmark.py         Question schema, provenance, validation
    ├── metrics.py           ROUGE, full METEOR, chrF, pinned BERTScore
    ├── retrieval_metrics.py Recall@k, P@k, MRR, nDCG, evidence hit
    ├── ocr_eval.py          Grapheme-cluster CER, WER, sample sizing
    ├── faithfulness.py      Support score, abstention, human judgements
    ├── stats.py             Bootstrap CIs, paired tests, Holm, Cliff's delta
    ├── runner.py            Writes the one raw log
    ├── tables.py            Derives every table; enforces consistency
    └── ablations.py         Threshold, retrieval, chunking, encoder sweeps

frontend/              Kannada-first web interface with on-screen keyboard
benchmark/             Question set, annotation guide, OCR transcriptions
tests/                 Metric correctness and table-consistency tests
docs/                  Reproducibility notes and migration guide
```

---

## Development

```bash
make test     # 35 tests: metric definitions, retrieval scoring, table consistency
make lint     # ruff + mypy
```

The test suite deliberately includes a case that fails if the table consistency
check *cannot* detect an inconsistency — a check that can never fail is worth
nothing.

---

## Security note

Earlier revisions of this repository committed a `backend/.env` file containing a
live API key. It has been removed and the key revoked. `.env` is now gitignored
and the default pipeline needs no key at all. If you forked this repository
before that change, treat any key found in your fork's history as compromised.

---

## Limitations

- The corpus is small and curated. Nothing here is tested against noisy,
  conflicting or adversarially incorrect sources.
- OCR quality is measured on a sample; conjunct-heavy and degraded pages remain
  the weakest part of ingestion.
- The retrieval encoder was not trained on Kannada heritage text. Its suitability
  is an empirical result from the encoder ablation, not an assumption.
- Dense retrieval alone matches proper nouns and inscription identifiers poorly.
  Hybrid BM25 + dense retrieval is the obvious next step and is not implemented.

---

## Citation

See [`CITATION.cff`](CITATION.cff). Paper citation will be added on publication.

## License

MIT — see [LICENSE](LICENSE). The heritage documents under `data/` may carry
separate rights.

## Acknowledgements

Ollama, FAISS, sentence-transformers, PyMuPDF and Tesseract, and the archivists
and scholars who documented Karnataka's heritage in Kannada.

<div align="center">

*ಕನ್ನಡ ನಾಡಿನ ಪರಂಪರೆಗಾಗಿ*

</div>
