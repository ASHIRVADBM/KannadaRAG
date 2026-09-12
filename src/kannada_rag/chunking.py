"""Recursive character chunking with inherited provenance.

Implemented directly rather than delegating to a framework splitter. Three
reasons, all of them reproducibility concerns:

1. The separator hierarchy is Kannada-specific (danda and double danda are
   sentence separators alongside the Latin full stop) and needs to be visible
   in the paper, not buried in a dependency's defaults.
2. Framework splitters change their boundary behaviour between releases, which
   silently changes the index and therefore every downstream number.
3. Every chunk must inherit ``source_file`` and ``page`` so that a generated
   answer can be traced to a page of a source document. Provenance is the
   mechanism behind the paper's traceability claim and cannot be optional.

The module also reports the *realised* chunk-length distribution in characters
and tokens, so the paper can state what chunk sizes actually occurred instead
of quoting the nominal configuration value.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Sequence

from .config import ChunkConfig
from .ingest import PageRecord
from .text import tokenise_for_metrics

__all__ = ["Chunk", "chunk_pages", "split_text", "chunk_length_stats"]


@dataclass
class Chunk:
    """A retrievable unit of text with full provenance back to a source page."""

    chunk_id: str
    text: str
    source_file: str
    page: int
    used_ocr: bool = False
    char_start: int = 0
    char_len: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "page": self.page,
            "used_ocr": self.used_ocr,
            "char_start": self.char_start,
            "char_len": self.char_len,
            **self.metadata,
        }

    @property
    def citation(self) -> str:
        return f"{self.source_file} p.{self.page}"


def _split_recursive(text: str, separators: Sequence[str], size: int) -> list[str]:
    """Split ``text`` into pieces of at most ``size`` characters.

    Tries separators in priority order. A piece that is still too long after
    the highest-priority separator is re-split with the next one down, so
    paragraph boundaries are preferred over line boundaries, which are
    preferred over sentence boundaries, and so on to a hard character cut.
    """
    if len(text) <= size:
        return [text] if text else []

    if not separators:
        return [text[i : i + size] for i in range(0, len(text), size)]

    separator, *rest = separators

    if separator == "":
        return [text[i : i + size] for i in range(0, len(text), size)]

    parts = text.split(separator)
    if len(parts) == 1:
        return _split_recursive(text, rest, size)

    # Re-attach the separator to every part but the last, so no characters are
    # lost and offsets remain meaningful.
    parts = [p + separator for p in parts[:-1]] + [parts[-1]]

    pieces: list[str] = []
    buffer = ""
    for part in parts:
        if len(part) > size:
            if buffer:
                pieces.append(buffer)
                buffer = ""
            pieces.extend(_split_recursive(part, rest, size))
            continue

        if len(buffer) + len(part) <= size:
            buffer += part
        else:
            if buffer:
                pieces.append(buffer)
            buffer = part

    if buffer:
        pieces.append(buffer)

    return [p for p in pieces if p.strip()]


def _apply_overlap(pieces: list[str], overlap: int) -> list[str]:
    """Prepend the tail of each piece to its successor.

    Overlap *reduces* the chance that a fact spanning a chunk boundary is lost
    from both chunks. It does not guarantee preservation: a statement longer
    than ``overlap`` characters that straddles a boundary can still be split
    across two chunks with neither containing it in full. The paper should
    state the mechanism in those terms, and the empirical effect of varying
    ``overlap`` is measured by the ablation suite.
    """
    if overlap <= 0 or len(pieces) < 2:
        return pieces

    out = [pieces[0]]
    for prev, current in zip(pieces, pieces[1:]):
        tail = prev[-overlap:] if len(prev) > overlap else prev
        out.append(tail + current)
    return out


def split_text(text: str, cfg: ChunkConfig) -> list[str]:
    """Split a single string according to the configured chunking policy."""
    pieces = _split_recursive(text, list(cfg.separators), cfg.chunk_size)
    return _apply_overlap(pieces, cfg.chunk_overlap)


def chunk_pages(pages: Sequence[PageRecord], cfg: ChunkConfig) -> list[Chunk]:
    """Chunk every page, carrying provenance into each resulting chunk."""
    chunks: list[Chunk] = []
    for page in pages:
        if not page.text.strip():
            continue
        offset = 0
        for i, piece in enumerate(split_text(page.text, cfg)):
            chunks.append(
                Chunk(
                    chunk_id=f"{page.source_file}#p{page.page}#c{i}",
                    text=piece,
                    source_file=page.source_file,
                    page=page.page,
                    used_ocr=page.used_ocr,
                    char_start=offset,
                    char_len=len(piece),
                )
            )
            offset += max(len(piece) - cfg.chunk_overlap, 1)
    return chunks


def chunk_length_stats(chunks: Sequence[Chunk]) -> dict:
    """Realised chunk-length distribution, in characters and metric tokens.

    Reported in the paper in place of the nominal chunk size, because the
    nominal size is an upper bound that the separator hierarchy rarely
    saturates.
    """
    if not chunks:
        return {}

    char_lengths = [len(c.text) for c in chunks]
    token_lengths = [len(tokenise_for_metrics(c.text)) for c in chunks]

    def describe(values: list[int]) -> dict:
        ordered = sorted(values)
        return {
            "n": len(ordered),
            "mean": round(statistics.mean(ordered), 1),
            "median": ordered[len(ordered) // 2],
            "stdev": round(statistics.pstdev(ordered), 1) if len(ordered) > 1 else 0.0,
            "min": ordered[0],
            "p25": ordered[int(0.25 * (len(ordered) - 1))],
            "p75": ordered[int(0.75 * (len(ordered) - 1))],
            "max": ordered[-1],
        }

    return {
        "characters": describe(char_lengths),
        "tokens": describe(token_lengths),
        "n_chunks": len(chunks),
        "n_ocr_derived_chunks": sum(1 for c in chunks if c.used_ocr),
    }
