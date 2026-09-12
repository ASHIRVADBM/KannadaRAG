"""Single source of truth for every tunable parameter in the pipeline.

Every number that appears in the manuscript is defined here exactly once and
imported everywhere else. Nothing downstream is allowed to hard-code a chunk
size, a threshold or a model name. This is deliberate: the previous version of
this codebase carried three different chunk sizes (500 in the paper, 1000 in
the backend, 800-1000 in the README), which is precisely the class of
inconsistency that makes published numbers unreproducible.

Configs are serialisable. Every evaluation artefact written by
``kannada_rag.eval`` embeds the exact ``PipelineConfig`` used to produce it, so
any table in the paper can be traced back to the settings that generated it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "IngestConfig",
    "ChunkConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "GenerationConfig",
    "PipelineConfig",
    "DEFAULT",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class IngestConfig:
    """PDF ingestion and OCR fallback.

    The OCR fallback is a *hybrid* strategy: native text is extracted first with
    PyMuPDF, and a page is re-rendered and sent to Tesseract only when the
    native layer yields fewer than ``ocr_char_threshold`` characters. The OCR
    output replaces the native text only if it is strictly longer, which
    protects mixed-content pages from being degraded.
    """

    ocr_char_threshold: int = 50
    """Native-text character count below which a page is treated as scanned."""

    ocr_dpi: int = 200
    """Rasterisation DPI for Tesseract. Follows the >=200 DPI recommendation
    for Kannada script reported by Upadhyay et al. (2024)."""

    tesseract_lang: str = "kan"
    tesseract_psm: int = 3
    """Page segmentation mode 3 = fully automatic, no orientation detection."""

    keep_ocr_only_if_longer: bool = True
    normalise_unicode: bool = True
    unicode_form: Literal["NFC", "NFKC"] = "NFC"


@dataclass(frozen=True)
class ChunkConfig:
    """Recursive character chunking.

    ``chunk_size`` and ``chunk_overlap`` are *character* counts, not tokens.
    The realised token-length distribution is reported by
    ``scripts/report_chunk_stats.py`` rather than asserted.
    """

    chunk_size: int = 500
    chunk_overlap: int = 50
    separators: tuple[str, ...] = (
        "\n\n",       # paragraph
        "\n",         # line
        "॥",     # Kannada/Devanagari double danda
        "।",     # danda
        ". ",         # Latin sentence boundary
        " ",          # word
        "",           # character fallback
    )


@dataclass(frozen=True)
class EmbeddingConfig:
    """Sentence embedding model used for *retrieval*.

    Kept deliberately separate from the BERTScore checkpoint in
    :class:`~kannada_rag.eval.metrics.MetricConfig`. Conflating the retrieval
    encoder with the evaluation encoder was a documented weakness of the
    earlier write-up; they are different models serving different purposes and
    are never interchanged.
    """

    model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    revision: str | None = None
    """Pin a git revision of the checkpoint for exact reproducibility."""

    normalise_embeddings: bool = True
    """L2-normalise before indexing. Required for inner product == cosine."""

    batch_size: int = 32
    device: str | None = None  # None -> auto-detect


@dataclass(frozen=True)
class RetrievalConfig:
    """Dense retrieval over a FAISS index.

    **Metric semantics.** The index is ``IndexFlatIP`` over L2-normalised
    vectors, so the raw index score is *cosine similarity* in ``[-1, 1]``,
    where higher is better. ``sim_threshold`` is therefore a similarity
    *floor*, not a distance ceiling. (An earlier description of this system
    called the same 0.55 value a "MIN SCORE" on an ``IndexFlatL2`` distance,
    which inverted the comparison; the code has always been cosine.)

    **Adaptive K.** ``k_max`` candidates are retrieved, then two rules decide
    how many survive:

    1. *absolute* -- keep chunks with ``sim >= sim_threshold``;
    2. *relative* -- keep chunks with ``sim >= relative_floor * top_sim``.

    A chunk must satisfy both. At most ``k_max`` and at least ``k_min`` chunks
    are passed to the generator, unless nothing clears the absolute floor, in
    which case the system abstains. This replaces the earlier practice of
    choosing K = 3 or 5 by hand per query, which could not be reproduced.
    """

    k_max: int = 5
    k_min: int = 1
    sim_threshold: float = 0.55
    relative_floor: float = 0.85
    """Fraction of the top score a chunk must reach to be retained."""

    abstain_when_empty: bool = True
    """Return a calibrated 'not in corpus' response rather than generating."""

    index_type: Literal["flat_ip"] = "flat_ip"


@dataclass(frozen=True)
class GenerationConfig:
    """Local or hosted LLM generation under *matched* decoding settings.

    The same decoding parameters are applied to every model in every
    condition. Comparing a locally deployed model under one temperature
    against a hosted model under its own defaults does not isolate the effect
    of retrieval; matched decoding is what makes the RAG / no-RAG contrast
    interpretable.
    """

    provider: Literal["ollama", "groq", "echo"] = "ollama"
    model: str = "gemma3:4b"
    temperature: float = 0.3
    max_tokens: int = 150
    top_p: float = 1.0
    seed: int | None = 0
    timeout_s: float = 180.0
    ollama_host: str = "http://localhost:11434"
    num_repeats: int = 3
    """Independent generations per question, for variance reporting."""


@dataclass(frozen=True)
class PipelineConfig:
    """The complete, hashable description of one experimental condition."""

    ingest: IngestConfig = field(default_factory=IngestConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    use_retrieval: bool = True
    """False runs the identical prompt and decoding without retrieved context.
    This is the controlled ablation behind every 'RAG helps' claim."""

    data_dir: Path = REPO_ROOT / "data"
    index_dir: Path = REPO_ROOT / "vector_store"
    results_dir: Path = REPO_ROOT / "results"
    random_seed: int = 20240101

    # ---------------------------------------------------------------- helpers

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("data_dir", "index_dir", "results_dir"):
            out[key] = str(out[key])
        out["chunk"]["separators"] = list(out["chunk"]["separators"])
        return out

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=False)

    @property
    def index_fingerprint(self) -> str:
        """Stable hash of every setting that affects the *index* contents.

        Changing chunk size or the embedding model must invalidate a cached
        index. Changing the LLM must not. The fingerprint is written alongside
        the index and checked on load, so a stale index can never silently
        serve an experiment it does not belong to.
        """
        payload = {
            "ingest": asdict(self.ingest),
            "chunk": {**asdict(self.chunk), "separators": list(self.chunk.separators)},
            "embedding": asdict(self.embedding),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def condition_id(self) -> str:
        """Short human-readable identifier used in filenames and table rows."""
        mode = "rag" if self.use_retrieval else "norag"
        model = self.generation.model.replace(":", "-").replace("/", "-")
        return f"{model}__{mode}__k{self.retrieval.k_max}__c{self.chunk.chunk_size}"

    def with_(self, **overrides: Any) -> "PipelineConfig":
        """Return a copy with top-level fields replaced (for sweeps)."""
        return replace(self, **overrides)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineConfig":
        payload = dict(payload)
        chunk = dict(payload.get("chunk", {}))
        if "separators" in chunk:
            chunk["separators"] = tuple(chunk["separators"])
        return cls(
            ingest=IngestConfig(**payload.get("ingest", {})),
            chunk=ChunkConfig(**chunk),
            embedding=EmbeddingConfig(**payload.get("embedding", {})),
            retrieval=RetrievalConfig(**payload.get("retrieval", {})),
            generation=GenerationConfig(**payload.get("generation", {})),
            use_retrieval=payload.get("use_retrieval", True),
            data_dir=Path(payload.get("data_dir", REPO_ROOT / "data")),
            index_dir=Path(payload.get("index_dir", REPO_ROOT / "vector_store")),
            results_dir=Path(payload.get("results_dir", REPO_ROOT / "results")),
            random_seed=payload.get("random_seed", 20240101),
        )

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


DEFAULT = PipelineConfig()
