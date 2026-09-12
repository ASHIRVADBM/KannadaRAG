"""FAISS vector index over L2-normalised multilingual sentence embeddings.

**Metric semantics, stated once and enforced everywhere.** The index is a
``faiss.IndexFlatIP`` built over vectors that have been L2-normalised at
insertion time and again at query time. For unit vectors, inner product is
exactly cosine similarity, so every score this module returns lies in
``[-1, 1]`` and *higher is better*. The retrieval threshold is consequently a
similarity floor.

This is spelled out because the same 0.55 constant has previously been
described as a distance ceiling on an ``IndexFlatL2`` index, which reverses the
direction of the comparison. Any reader checking the threshold against the
code needs the two to agree.

The index stores chunk text alongside the vectors. An index that stores only
metadata cannot return the passages the generator needs, and a retrieval layer
that returns filenames instead of text is not doing retrieval-augmented
generation at all.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .chunking import Chunk
from .config import EmbeddingConfig, RetrievalConfig

logger = logging.getLogger(__name__)

__all__ = ["Embedder", "VectorIndex", "ScoredChunk"]


@dataclass
class ScoredChunk:
    """A retrieved chunk together with its cosine similarity to the query."""

    chunk: Chunk
    score: float
    rank: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk.chunk_id,
            "score": round(self.score, 6),
            "rank": self.rank,
            "source_file": self.chunk.source_file,
            "page": self.chunk.page,
            "used_ocr": self.chunk.used_ocr,
            "text": self.chunk.text,
        }


class Embedder:
    """Wraps the sentence-transformer encoder used for *retrieval only*."""

    def __init__(self, cfg: EmbeddingConfig):
        from sentence_transformers import SentenceTransformer

        self.cfg = cfg
        kwargs = {}
        if cfg.revision:
            kwargs["revision"] = cfg.revision
        if cfg.device:
            kwargs["device"] = cfg.device
        logger.info("Loading retrieval encoder %s", cfg.model_name)
        self.model = SentenceTransformer(cfg.model_name, **kwargs)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            batch_size=self.cfg.batch_size,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 256,
        ).astype(np.float32)
        if self.cfg.normalise_embeddings:
            vectors = l2_normalise(vectors)
        return vectors


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation with a guard against zero vectors."""
    vectors = np.atleast_2d(vectors).astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-10, None)


class VectorIndex:
    """Exact cosine-similarity index with persisted text and provenance.

    Persistence layout under ``index_dir``::

        faiss.index       the FAISS IndexFlatIP
        chunks.jsonl      chunk text + provenance, one JSON object per line,
                          in the same order as the vectors
        manifest.json     fingerprint of the config that produced the index

    The manifest is checked on load. An index whose fingerprint does not match
    the current configuration is refused rather than used, so a stale index
    built with a different chunk size can never silently serve an experiment.
    """

    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks: list[Chunk] = []

    # ------------------------------------------------------------------ build

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != vectors.shape[0]:
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} vs {vectors.shape[0]}"
            )
        self.index.add(l2_normalise(vectors))
        self.chunks.extend(chunks)

    # ------------------------------------------------------------------ query

    def search(self, query_vector: np.ndarray, k: int) -> list[ScoredChunk]:
        """Return the ``k`` nearest chunks by cosine similarity, unfiltered.

        Thresholding is the retriever's job, not the index's. Keeping them
        separate means the evaluation harness can inspect the full ranked list
        (needed for Recall@K, MRR and nDCG) while the serving path still
        applies the abstention threshold.
        """
        if not self.chunks:
            return []

        query = l2_normalise(np.asarray(query_vector, dtype=np.float32))
        k = min(k, len(self.chunks))
        scores, indices = self.index.search(query, k)

        results: list[ScoredChunk] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(
                ScoredChunk(chunk=self.chunks[int(idx)], score=float(score), rank=rank)
            )
        return results

    # ------------------------------------------------------------- persistence

    def save(self, index_dir: str | Path, fingerprint: str, extra: dict | None = None) -> None:
        import faiss

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / "faiss.index"))

        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for chunk in self.chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        manifest = {
            "fingerprint": fingerprint,
            "dim": self.dim,
            "n_chunks": len(self.chunks),
            "metric": "cosine (inner product over L2-normalised vectors)",
            **(extra or {}),
        }
        (index_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Wrote index with %d chunks to %s", len(self.chunks), index_dir)

    @classmethod
    def load(cls, index_dir: str | Path, fingerprint: str | None = None) -> "VectorIndex":
        import faiss

        index_dir = Path(index_dir)
        manifest_path = index_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No index manifest at {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if fingerprint is not None and manifest.get("fingerprint") != fingerprint:
            raise ValueError(
                "Index fingerprint mismatch. The persisted index was built with a "
                "different ingestion, chunking or embedding configuration "
                f"({manifest.get('fingerprint')} != {fingerprint}). Rebuild it with "
                "`python -m kannada_rag.cli build --force`."
            )

        instance = cls.__new__(cls)
        instance.dim = manifest["dim"]
        instance.index = faiss.read_index(str(index_dir / "faiss.index"))

        instance.chunks = []
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                payload = json.loads(line)
                instance.chunks.append(
                    Chunk(
                        chunk_id=payload["chunk_id"],
                        text=payload["text"],
                        source_file=payload["source_file"],
                        page=payload["page"],
                        used_ocr=payload.get("used_ocr", False),
                        char_start=payload.get("char_start", 0),
                        char_len=payload.get("char_len", 0),
                    )
                )

        if instance.index.ntotal != len(instance.chunks):
            raise ValueError(
                f"Corrupt index: {instance.index.ntotal} vectors but "
                f"{len(instance.chunks)} chunks."
            )

        logger.info("Loaded index with %d chunks from %s", len(instance.chunks), index_dir)
        return instance

    def __len__(self) -> int:
        return len(self.chunks)
