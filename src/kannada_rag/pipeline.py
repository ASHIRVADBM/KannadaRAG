"""The end-to-end pipeline, in both the grounded and ungrounded conditions.

One class serves both conditions. ``use_retrieval`` toggles whether the
retrieved context is placed in the prompt; everything else -- the model, the
decoding settings, the sentence-length instruction, the language guard -- is
held identical. That is what makes the difference between the two conditions
attributable to retrieval.

The pipeline also separates the two things previously collapsed into a single
"average query time": embedding plus retrieval latency, and generation
latency. They scale differently with hardware and only the second is affected
by the choice of model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import chunk_length_stats, chunk_pages
from .config import PipelineConfig
from .generation import GenerationOutput, get_provider
from .index import Embedder, VectorIndex
from .ingest import ingest_corpus
from .prompts import ABSTENTION_RESPONSE, NON_KANNADA_RESPONSE, build_prompt
from .retriever import RetrievalResult, Retriever
from .text import is_predominantly_kannada

logger = logging.getLogger(__name__)

__all__ = ["Answer", "RAGPipeline", "build_index"]


@dataclass
class Answer:
    """A complete, auditable record of answering one question."""

    question: str
    answer: str
    status: str
    """One of: ok, abstained, rejected_language, error."""

    retrieval: RetrievalResult | None = None
    generation: GenerationOutput | None = None
    prompt: str | None = None

    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0

    condition: str = ""
    used_retrieval: bool = True
    sources: list[dict] = field(default_factory=list)

    @property
    def total_latency_s(self) -> float:
        return self.retrieval_latency_s + self.generation_latency_s

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "status": self.status,
            "condition": self.condition,
            "used_retrieval": self.used_retrieval,
            "retrieval_latency_s": round(self.retrieval_latency_s, 4),
            "generation_latency_s": round(self.generation_latency_s, 4),
            "total_latency_s": round(self.total_latency_s, 4),
            "sources": self.sources,
            "retrieval": self.retrieval.to_dict() if self.retrieval else None,
            "generation": self.generation.to_dict() if self.generation else None,
            "prompt": self.prompt,
        }


def build_index(cfg: PipelineConfig, force: bool = False) -> VectorIndex:
    """Build or load the vector index for ``cfg``.

    A cached index is reused only when its fingerprint matches the current
    ingestion, chunking and embedding configuration. Changing the chunk size
    therefore forces a rebuild automatically, which is the behaviour the
    ablation sweep depends on.
    """
    embedder = Embedder(cfg.embedding)

    if not force:
        try:
            return VectorIndex.load(cfg.index_dir, cfg.index_fingerprint)
        except (FileNotFoundError, ValueError) as exc:
            logger.info("Rebuilding index: %s", exc)

    report = ingest_corpus(cfg.data_dir, cfg.ingest)
    report.write(Path(cfg.index_dir) / "ingest_report.json")

    chunks = chunk_pages(report.pages, cfg.chunk)
    if not chunks:
        raise RuntimeError(f"Corpus at {cfg.data_dir} produced no chunks.")

    stats = chunk_length_stats(chunks)
    logger.info("Chunk statistics: %s", stats)

    vectors = embedder.encode([c.text for c in chunks])

    index = VectorIndex(embedder.dim)
    index.add(chunks, vectors)
    index.save(
        cfg.index_dir,
        cfg.index_fingerprint,
        extra={
            "chunk_stats": stats,
            "ingest_summary": report.summary(),
            "embedding_model": cfg.embedding.model_name,
            "config": cfg.to_dict(),
        },
    )
    return index


class RAGPipeline:
    """Serving and evaluation entry point."""

    def __init__(self, cfg: PipelineConfig, index: VectorIndex | None = None):
        self.cfg = cfg
        self.embedder = Embedder(cfg.embedding)
        self.index = index if index is not None else build_index(cfg)
        self.retriever = Retriever(self.embedder, self.index, cfg.retrieval)
        self.provider = get_provider(cfg.generation)

    # ------------------------------------------------------------------ answer

    def answer(self, question: str, enforce_language_guard: bool = True) -> Answer:
        condition = self.cfg.condition_id

        if enforce_language_guard and not is_predominantly_kannada(question):
            return Answer(
                question=question,
                answer=NON_KANNADA_RESPONSE,
                status="rejected_language",
                condition=condition,
                used_retrieval=self.cfg.use_retrieval,
            )

        # -------------------------------------------------- ungrounded control
        if not self.cfg.use_retrieval:
            prompt = build_prompt(question, context=None)
            generation = self.provider.generate(prompt)
            return Answer(
                question=question,
                answer=generation.text,
                status="error" if generation.error else "ok",
                generation=generation,
                prompt=prompt,
                generation_latency_s=generation.latency_s,
                condition=condition,
                used_retrieval=False,
            )

        # ------------------------------------------------------ grounded path
        start = time.perf_counter()
        retrieval = self.retriever.retrieve(question)
        retrieval_latency = time.perf_counter() - start

        if retrieval.abstained:
            return Answer(
                question=question,
                answer=ABSTENTION_RESPONSE,
                status="abstained",
                retrieval=retrieval,
                retrieval_latency_s=retrieval_latency,
                condition=condition,
                used_retrieval=True,
            )

        prompt = build_prompt(question, context=retrieval.context)
        generation = self.provider.generate(prompt)

        return Answer(
            question=question,
            answer=generation.text,
            status="error" if generation.error else "ok",
            retrieval=retrieval,
            generation=generation,
            prompt=prompt,
            retrieval_latency_s=retrieval_latency,
            generation_latency_s=generation.latency_s,
            condition=condition,
            used_retrieval=True,
            sources=[
                {
                    "chunk_id": sc.chunk.chunk_id,
                    "citation": sc.chunk.citation,
                    "page": sc.chunk.page,
                    "source_file": sc.chunk.source_file,
                    "score": round(sc.score, 4),
                    "used_ocr": sc.chunk.used_ocr,
                }
                for sc in retrieval.selected
            ],
        )
