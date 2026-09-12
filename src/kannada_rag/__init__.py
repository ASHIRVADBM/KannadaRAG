"""Kannada-centric retrieval-augmented generation over Karnataka heritage documents.

Public surface::

    from kannada_rag import PipelineConfig, RAGPipeline, build_index

Everything else is importable but internal; module docstrings explain the
design decision each one encodes.
"""

from .config import (
    DEFAULT,
    ChunkConfig,
    EmbeddingConfig,
    GenerationConfig,
    IngestConfig,
    PipelineConfig,
    RetrievalConfig,
)
from .pipeline import Answer, RAGPipeline, build_index

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "DEFAULT",
    "IngestConfig",
    "ChunkConfig",
    "EmbeddingConfig",
    "RetrievalConfig",
    "GenerationConfig",
    "PipelineConfig",
    "RAGPipeline",
    "Answer",
    "build_index",
]
