"""Hybrid PDF ingestion with per-page OCR provenance.

The hybrid strategy (native extraction, OCR fallback for image-only pages) is
one of the Kannada-specific claims of this system, so the ingestion layer is
instrumented to make that claim measurable rather than asserted:

* every page records whether OCR ran, why it ran, and how many characters each
  extraction path produced;
* an :class:`IngestReport` aggregates those records into the counts a reader
  needs (how many pages required OCR, what fraction of the corpus that is);
* :mod:`kannada_rag.eval.ocr_eval` consumes the same records to compute CER and
  WER against manual transcriptions of a sampled subset.

Nothing here silently discards information. A page whose OCR output is
rejected still records the rejected candidate's length, so the decision rule
itself can be audited.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from .config import IngestConfig
from .text import kannada_ratio, normalise_for_index

logger = logging.getLogger(__name__)

__all__ = ["PageRecord", "IngestReport", "extract_pdf", "ingest_corpus"]


@dataclass
class PageRecord:
    """One PDF page, with a complete record of how its text was obtained."""

    source_file: str
    page: int
    text: str

    used_ocr: bool = False
    native_chars: int = 0
    ocr_chars: int = 0
    ocr_attempted: bool = False
    ocr_rejected: bool = False
    """True when OCR ran but its output was not longer than the native text."""

    kannada_ratio: float = 0.0

    @property
    def doc_id(self) -> str:
        return f"{self.source_file}#p{self.page}"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["doc_id"] = self.doc_id
        return payload


@dataclass
class IngestReport:
    """Corpus-level ingestion statistics, written next to the index."""

    pages: list[PageRecord] = field(default_factory=list)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def n_ocr_pages(self) -> int:
        return sum(1 for p in self.pages if p.used_ocr)

    @property
    def n_ocr_attempted(self) -> int:
        return sum(1 for p in self.pages if p.ocr_attempted)

    @property
    def n_ocr_rejected(self) -> int:
        return sum(1 for p in self.pages if p.ocr_rejected)

    @property
    def n_empty_pages(self) -> int:
        return sum(1 for p in self.pages if not p.text.strip())

    @property
    def ocr_page_fraction(self) -> float:
        return self.n_ocr_pages / self.n_pages if self.n_pages else 0.0

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

    def summary(self) -> dict:
        return {
            "n_pages": self.n_pages,
            "n_ocr_attempted": self.n_ocr_attempted,
            "n_ocr_used": self.n_ocr_pages,
            "n_ocr_rejected": self.n_ocr_rejected,
            "ocr_page_fraction": round(self.ocr_page_fraction, 4),
            "n_empty_pages": self.n_empty_pages,
            "total_chars": self.total_chars,
            "mean_chars_per_page": round(self.total_chars / self.n_pages, 1)
            if self.n_pages
            else 0.0,
            "mean_kannada_ratio": round(
                sum(p.kannada_ratio for p in self.pages) / self.n_pages, 4
            )
            if self.n_pages
            else 0.0,
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.summary(),
            "pages": [p.to_dict() for p in self.pages],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path


def _render_page_to_image(page, dpi: int):
    """Rasterise a PyMuPDF page for OCR. Imported lazily so that the metric and
    statistics modules can be used without the PDF/OCR stack installed."""
    import io

    from PIL import Image

    pixmap = page.get_pixmap(dpi=dpi)
    return Image.open(io.BytesIO(pixmap.tobytes("png")))


def extract_pdf(pdf_path: str | Path, cfg: IngestConfig) -> list[PageRecord]:
    """Extract every page of one PDF using the hybrid strategy.

    Native text is taken first. A page is sent to OCR only when the native
    layer produced fewer than ``cfg.ocr_char_threshold`` characters, which is
    the operational definition of "scanned page" used throughout this work.
    """
    import fitz  # PyMuPDF

    pdf_path = Path(pdf_path)
    records: list[PageRecord] = []

    with fitz.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf):
            native = page.get_text("text").strip()
            record = PageRecord(
                source_file=pdf_path.name,
                page=i + 1,
                text=native,
                native_chars=len(native),
            )

            if len(native) < cfg.ocr_char_threshold:
                record.ocr_attempted = True
                try:
                    import pytesseract

                    image = _render_page_to_image(page, cfg.ocr_dpi)
                    ocr_text = pytesseract.image_to_string(
                        image,
                        lang=cfg.tesseract_lang,
                        config=f"--psm {cfg.tesseract_psm}",
                    ).strip()
                    record.ocr_chars = len(ocr_text)

                    longer = len(ocr_text) > len(native)
                    if longer or not cfg.keep_ocr_only_if_longer:
                        record.text = ocr_text
                        record.used_ocr = True
                    else:
                        record.ocr_rejected = True
                except Exception as exc:  # pragma: no cover - environment dependent
                    logger.warning(
                        "OCR failed on %s page %d: %s", pdf_path.name, i + 1, exc
                    )
                    record.ocr_rejected = True

            if cfg.normalise_unicode:
                record.text = normalise_for_index(record.text, cfg.unicode_form)

            record.kannada_ratio = kannada_ratio(record.text)
            records.append(record)

    return records


def ingest_corpus(
    data_dir: str | Path,
    cfg: IngestConfig,
    patterns: Sequence[str] = ("*.pdf", "*.PDF"),
) -> IngestReport:
    """Ingest every PDF under ``data_dir`` into a single :class:`IngestReport`."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {data_dir}")

    paths = sorted({p for pattern in patterns for p in data_dir.glob(pattern)})
    if not paths:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    report = IngestReport()
    for path in paths:
        logger.info("Ingesting %s", path.name)
        report.pages.extend(extract_pdf(path, cfg))

    logger.info("Ingested %s", report.summary())
    return report


def iter_nonempty(report: IngestReport) -> Iterator[PageRecord]:
    for page in report.pages:
        if page.text.strip():
            yield page
