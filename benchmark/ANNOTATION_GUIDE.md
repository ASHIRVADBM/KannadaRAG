# Benchmark construction and annotation guide

This document specifies how the Kannada heritage QA benchmark is built. It
exists so that the question set can be described in a paper in terms a reviewer
can check: who wrote the questions, whether they were independent of system
development, who validated them, and what counts as the evidence for an answer.

---

## 1. Why the original 25-question set was not enough

The first version of this benchmark had three properties that limit what can be
concluded from it, and all three are addressed below.

| Limitation | Consequence | Fix |
|---|---|---|
| 25 questions, all from one 62-page document | Confidence intervals wider than most between-model differences; no test of behaviour on noisy or conflicting sources | Target **150+** questions across **multiple documents** (§2) |
| No record of who wrote questions or whether they were independent | Cannot rule out questions written to suit the system | `author` and `author_independent` per question (§3) |
| Every question answerable from the corpus | No way to tell a grounded system from a fluent one; abstention untested | **Unanswerable tier**, 20% of the set (§4) |
| No annotated evidence passages | Retrieval quality unmeasurable; only end-to-end answer quality observable | `gold_pages` per question (§5) |

---

## 2. Size and composition

Target composition, to be scaled as annotator time allows:

| Tier | Share | n at 150 | What it tests |
|---|---|---|---|
| Easy | 25% | 38 | Single-site factual recall from one passage |
| Medium | 25% | 38 | Interpretive synthesis across 2–3 passages |
| Hard | 20% | 30 | Comparative or multi-hop reasoning across sites or dynasties |
| Unanswerable | 20% | 30 | Abstention (§4) |
| Adversarial | 10% | 14 | Near-miss distractors (§6) |

**Minimum viable set.** If 150 is not achievable, 60 questions with the tier
proportions above and full gold-evidence annotation is substantially more
informative than 150 without annotation. Retrieval metrics and abstention
behaviour are what the evaluation currently lacks, not raw question count.

**Sample size.** To detect a 0.05 difference in BERT-F1 between two models with
80% power at α = 0.05, given the per-question standard deviation observed in a
pilot run, use:

```bash
python scripts/power_analysis.py --effect 0.05 --power 0.8
```

Report the result. If the benchmark is underpowered for the differences being
claimed, say so in the paper rather than reporting the difference as a finding.

---

## 3. Question authorship and independence

Each question records:

```json
"author": "annotator_03",
"author_independent": true,
"validated_by": ["expert_01", "expert_02"]
```

`author_independent` is **true** only when the author:

1. took no part in writing, tuning or configuring the system;
2. had no access to system outputs when writing the question;
3. wrote the question from the source document alone.

An author who has seen the system answer a question, and then writes a question
resembling it, is not independent — even in good faith. Record this honestly;
the proportion is reported in the paper and a mixed set with an accurate count
is far more credible than a uniform claim of independence.

**Recommended procedure.** Two annotators who are Kannada speakers and not
system developers read assigned page ranges and write questions from them. A
third person, a heritage or Kannada-language expert, validates each question and
its reference answers. The system is not consulted at any stage.

---

## 4. Unanswerable questions

Two kinds, and the mix matters:

**Out of domain** — plausible heritage questions about material the corpus does
not cover (a Karnataka site absent from the document, or a site in another
state). These are easy to write and catch gross failures.

**In domain, absent detail** — questions about a site the corpus *does* cover,
asking for a specific fact it does not contain: a date not given, a measurement
not recorded, a recent conservation report. These are the demanding cases. A
model with strong parametric knowledge of Hampi will be tempted to supply an
answer from memory, and that is exactly the hallucination the retrieval
grounding is supposed to prevent.

Aim for roughly half of each. Scoring is automatic: the system should return
status `abstained`. Both abstention recall and the false-abstention rate on
answerable questions are reported — a system that declines everything scores
perfect abstention recall and is useless.

---

## 5. Gold evidence annotation

For each answerable question, record every page containing evidence needed to
answer it:

```json
"source_file": "HISTORICAL.pdf",
"gold_pages": [7, 14, 21]
```

**Pages, not chunk ids.** Chunk ids change whenever the chunking configuration
changes, which would invalidate the annotation every time the chunk-size
ablation runs. Page numbers are stable across re-chunking. Chunk ids may be
recorded additionally, but pages are the primary key.

**What counts as evidence.** A page is gold if a knowledgeable reader could
construct part of the reference answer from it. A page that merely mentions the
site without supporting the answer is not gold. When two pages contain the same
fact, annotate both — retrieving either is a success.

**Multi-hop questions.** A hard question typically has 2–4 gold pages. Recall@k
should then be read against the number of gold pages: retrieving 2 of 3 at k=5
is a partial success and the metric reflects that.

**Agreement.** Two annotators independently mark gold pages for at least 20% of
questions. Report Cohen's κ on the page-level binary judgement. Below κ = 0.6,
the annotation guideline needs sharpening before the retrieval numbers mean
anything.

---

## 6. Adversarial questions

Questions whose top-scoring passage is *not* the right passage. Typically built
by naming an entity that appears prominently in a passage that does not answer
the question — a temple that the corpus discusses mostly in the context of a
different dynasty, for instance.

These are the questions that separate embedding-space similarity from actual
relevance, and they are where a system with a similarity threshold should either
retrieve correctly or abstain, never confidently answer from the wrong passage.

---

## 7. Reference answers

* Between **one and three** accepted phrasings per question. The scorer takes
  the best match, so a correct answer is not penalised for choosing a different
  valid wording — a real effect in Kannada, where one fact admits many surface
  forms.
* Written **in Kannada**, in the register the system is expected to produce.
* Length matched to the tier: 1–2 sentences easy, 2–3 medium, 3–4 hard. Do not
  vary reference length between evaluation runs; every metric here is
  length-sensitive, and comparing a run scored against short references with one
  scored against long references is not a valid comparison.
* Every change to a reference answer increments `reference_version` for the
  whole file. The harness records the version in every result record and refuses
  to aggregate across versions.

---

## 8. OCR transcriptions

Separately, for OCR evaluation, transcribe a sample of pages by hand into
`benchmark/ocr_transcriptions.jsonl`:

```json
{"source_file": "HISTORICAL.pdf", "page": 12, "transcription": "…", "transcriber": "annotator_02"}
```

How many pages:

```bash
python -m kannada_rag.cli ocr --sample-size --expected-cer 0.12 --half-width 0.03 --population 62
```

Stratify the sample: include every page that required OCR (if there are few),
plus a random sample of native-extraction pages as a control. Report CER and WER
separately for the two groups — pooling them dilutes the OCR error rate towards
zero and hides the figure that matters.

---

## 9. Checklist before running the evaluation

```bash
python -m kannada_rag.cli evaluate --strict
```

`--strict` refuses to run unless every question has gold evidence, every
question is expert-validated, there are no duplicate ids, and all questions
share one reference version. Fix the benchmark rather than dropping the flag.
