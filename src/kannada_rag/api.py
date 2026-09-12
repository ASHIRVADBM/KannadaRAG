"""FastAPI service.

Design points worth noting:

* The index is built once at startup and held in memory. Queries never touch
  disk, which is what keeps retrieval latency in the low milliseconds.
* Responses carry their sources, with document name, page number, similarity
  score and whether the passage came from OCR. A heritage answer a reader
  cannot trace to a page is not much use, and exposing the OCR flag lets a
  reader discount a passage that may carry recognition errors.
* Abstention is a first-class response status, not an error. The system
  declining to answer is correct behaviour when the corpus does not contain
  the answer, and the UI renders it differently from a failure.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import REPO_ROOT, PipelineConfig
from .pipeline import RAGPipeline

logger = logging.getLogger(__name__)

_state: dict = {}


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    k: int | None = Field(None, ge=1, le=20)


class Source(BaseModel):
    citation: str
    source_file: str
    page: int
    score: float
    used_ocr: bool
    excerpt: str = ""


class AskResponse(BaseModel):
    status: str
    answer: str
    sources: list[Source] = []
    retrieval_latency_s: float = 0.0
    generation_latency_s: float = 0.0
    total_latency_s: float = 0.0
    model: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = PipelineConfig()
    logger.info("Starting pipeline (%s)", cfg.condition_id)
    _state["pipeline"] = RAGPipeline(cfg)
    _state["config"] = cfg
    logger.info("Ready: %d chunks indexed", len(_state["pipeline"].index))
    yield
    _state.clear()


app = FastAPI(
    title="Kannada Heritage RAG",
    description="Retrieval-augmented question answering over Karnataka heritage documents.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    # Tightened from the previous wildcard. Override for a real deployment.
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000", "null"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    pipeline: RAGPipeline | None = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")
    cfg = _state["config"]
    return {
        "status": "ok",
        "n_chunks": len(pipeline.index),
        "model": cfg.generation.model,
        "provider": cfg.generation.provider,
        "embedding_model": cfg.embedding.model_name,
        "index_fingerprint": cfg.index_fingerprint,
    }


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    pipeline: RAGPipeline | None = _state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "Pipeline not ready")

    answer = pipeline.answer(request.query)

    sources = []
    if answer.retrieval:
        for scored in answer.retrieval.selected:
            excerpt = scored.chunk.text.strip()
            sources.append(
                Source(
                    citation=scored.chunk.citation,
                    source_file=scored.chunk.source_file,
                    page=scored.chunk.page,
                    score=round(scored.score, 4),
                    used_ocr=scored.chunk.used_ocr,
                    excerpt=excerpt[:400] + ("..." if len(excerpt) > 400 else ""),
                )
            )

    return AskResponse(
        status=answer.status,
        answer=answer.answer,
        sources=sources,
        retrieval_latency_s=round(answer.retrieval_latency_s, 4),
        generation_latency_s=round(answer.generation_latency_s, 4),
        total_latency_s=round(answer.total_latency_s, 4),
        model=_state["config"].generation.model,
    )


_frontend = REPO_ROOT / "frontend"
if _frontend.exists():
    app.mount("/", StaticFiles(directory=str(_frontend), html=True), name="frontend")
