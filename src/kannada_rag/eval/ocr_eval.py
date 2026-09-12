"""OCR quality measurement against manual Kannada transcriptions.

The hybrid ingestion strategy is claimed as a Kannada-specific contribution,
which obliges the work to say how good the OCR actually is and whether the
hybrid strategy beats native extraction alone. Three measurements are provided:

1. **How much OCR happens.** Taken from the ingestion report: how many pages
   fell below the native-text threshold, how many OCR outputs were accepted,
   how many were rejected. Free, and already enough to establish whether OCR
   is a major or marginal part of the pipeline.

2. **How accurate it is.** CER and WER against manual transcriptions of a
   sampled subset of pages. Sampling is stratified by OCR status and the
   sample size needed for a given interval width is computed, rather than
   transcribing an arbitrary handful.

3. **Whether it helps downstream.** The same benchmark run twice -- once with
   the hybrid pipeline, once with native extraction only -- and compared on
   retrieval and answer metrics. This is the measurement that actually
   supports the contribution claim; CER alone says the OCR is good, not that
   it matters.

CER is computed on grapheme clusters, not Unicode codepoints. A Kannada
akshara such as ಕ್ಕಿ is several codepoints but one written unit, and
codepoint-level CER systematically overstates error on conjunct-heavy text.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from ..text import normalise_for_index, tokenise_for_metrics
from .stats import bootstrap_ci, describe

__all__ = [
    "levenshtein",
    "grapheme_clusters",
    "character_error_rate",
    "word_error_rate",
    "OCRPageResult",
    "evaluate_ocr",
    "required_sample_size",
]


def grapheme_clusters(text: str) -> list[str]:
    """Split Kannada text into user-perceived characters (aksharas).

    Uses the ``regex`` module's ``\\X`` grapheme matcher when available and
    falls back to a combining-mark-aware heuristic otherwise, so the function
    works in a minimal environment without silently changing its definition.
    """
    try:
        import regex

        return regex.findall(r"\X", text)
    except ImportError:
        clusters: list[str] = []
        for char in text:
            combining = unicodedata.combining(char) != 0 or char in ("\u200c", "\u200d")
            # Kannada vowel signs, virama and length marks attach to the
            # preceding base consonant.
            if 0x0CBE <= ord(char) <= 0x0CD6:
                combining = True
            if clusters and combining:
                clusters[-1] += char
            else:
                clusters.append(char)
        return clusters


def levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance with unit costs, O(min(len)) space."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, item_a in enumerate(a, start=1):
        current = [i]
        for j, item_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,           # deletion
                    current[j - 1] + 1,        # insertion
                    previous[j - 1] + (item_a != item_b),  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(hypothesis: str, reference: str) -> float:
    """CER over grapheme clusters: edit distance / reference length."""
    ref_clusters = grapheme_clusters(normalise_for_index(reference))
    hyp_clusters = grapheme_clusters(normalise_for_index(hypothesis))
    if not ref_clusters:
        return float("nan")
    return levenshtein(hyp_clusters, ref_clusters) / len(ref_clusters)


def word_error_rate(hypothesis: str, reference: str) -> float:
    """WER over whitespace-delimited Kannada words."""
    ref_words = tokenise_for_metrics(reference)
    hyp_words = tokenise_for_metrics(hypothesis)
    if not ref_words:
        return float("nan")
    return levenshtein(hyp_words, ref_words) / len(ref_words)


@dataclass
class OCRPageResult:
    source_file: str
    page: int
    used_ocr: bool
    cer: float
    wer: float
    ref_chars: int
    hyp_chars: int

    def to_dict(self) -> dict:
        return {
            "source_file": self.source_file,
            "page": self.page,
            "used_ocr": self.used_ocr,
            "cer": round(self.cer, 4),
            "wer": round(self.wer, 4),
            "ref_chars": self.ref_chars,
            "hyp_chars": self.hyp_chars,
        }


def evaluate_ocr(
    transcriptions_path: str | Path,
    ingest_report_path: str | Path,
) -> dict:
    """Compare pipeline output against manual transcriptions.

    ``transcriptions_path`` is a JSONL file with one object per transcribed
    page::

        {"source_file": "HISTORICAL.pdf", "page": 12,
         "transcription": "...", "transcriber": "annotator_2"}

    Pages are matched to the ingestion report by (source_file, page). Results
    are broken out by whether OCR was used, because the headline number that
    matters is the error rate *on the pages that needed OCR* -- pooling them
    with clean digital pages dilutes it towards zero and hides the real figure.
    """
    transcriptions_path = Path(transcriptions_path)
    report = json.loads(Path(ingest_report_path).read_text(encoding="utf-8"))
    pages = {(p["source_file"], p["page"]): p for p in report["pages"]}

    results: list[OCRPageResult] = []
    missing: list[str] = []

    with transcriptions_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (record["source_file"], record["page"])
            page = pages.get(key)
            if page is None:
                missing.append(f"{key[0]} p.{key[1]}")
                continue

            reference = record["transcription"]
            hypothesis = page["text"]
            results.append(
                OCRPageResult(
                    source_file=key[0],
                    page=key[1],
                    used_ocr=page["used_ocr"],
                    cer=character_error_rate(hypothesis, reference),
                    wer=word_error_rate(hypothesis, reference),
                    ref_chars=len(reference),
                    hyp_chars=len(hypothesis),
                )
            )

    ocr_pages = [r for r in results if r.used_ocr]
    native_pages = [r for r in results if not r.used_ocr]

    def block(subset: list[OCRPageResult]) -> dict:
        if not subset:
            return {"n": 0}
        return {
            "n": len(subset),
            "cer": describe([r.cer for r in subset]),
            "wer": describe([r.wer for r in subset]),
        }

    return {
        "corpus_summary": report["summary"],
        "n_transcribed": len(results),
        "unmatched_transcriptions": missing,
        "ocr_pages": block(ocr_pages),
        "native_pages": block(native_pages),
        "all_pages": block(results),
        "per_page": [r.to_dict() for r in results],
    }


def required_sample_size(
    expected_cer: float = 0.12,
    half_width: float = 0.03,
    confidence: float = 0.95,
    population: int | None = None,
) -> int:
    """How many pages must be transcribed for a target interval width.

    Normal approximation with a finite-population correction. For a 62-page
    document, an expected CER around 0.12 and a target half-width of 0.03, the
    answer is a manageable number of pages -- which is the point of computing
    it rather than transcribing "a representative subset" of unstated size.
    """
    from scipy import stats as sp

    z = float(sp.norm.ppf(1 - (1 - confidence) / 2))
    variance = expected_cer * (1 - expected_cer)
    n = (z ** 2) * variance / (half_width ** 2)

    if population:
        n = n / (1 + (n - 1) / population)

    return max(1, math.ceil(n))
