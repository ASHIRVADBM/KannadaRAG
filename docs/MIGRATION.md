# Migration from v1

v2 replaces `backend/` and `rag.ipynb` with an installable package under
`src/kannada_rag/`. This note records what changed and why, because several of
the changes are corrections to defects that affected published results.

## Corrected defects

### 1. The retriever passed filenames to the model instead of passage text

`backend/rag_core.py` built its prompt context from
`r["meta"].get("source", "")` — the *filename* of the source PDF. The vector
store's `search()` returned only scores and metadata and never carried the chunk
text, so the retrieved passages could not reach the generator at all.

The served system was therefore not performing retrieval-augmented generation:
the model received the string `HISTORICAL.pdf` as its entire context and
answered from parametric memory. The research notebook did not have this defect
(it indexed back into `chunks` by position), so notebook and backend behaved
differently.

**Fixed:** `VectorIndex` persists chunk text alongside the vectors in
`chunks.jsonl`, and `RetrievalResult.context` composes the passage text with
citations. `tests/` covers the path.

### 2. Index loading discarded the chunks

`build_index()` returned `chunks = [Doc("", {})]` on the cached-index path, so
after the first run there were no chunks in memory at all.

**Fixed:** chunks are reloaded from `chunks.jsonl` and the index refuses to load
if the vector count and chunk count disagree.

### 3. The language guard accepted any query containing one Kannada character

`is_kannada_text()` in the backend returned `True` if a single Kannada character
appeared anywhere. English queries containing one Kannada word passed the guard
and produced English answers, defeating the monolingual design. The notebook
used a 50% ratio; the backend did not.

**Fixed:** `is_predominantly_kannada()` applies the ratio test, with whitespace
excluded from the denominator, and the frontend previews the ratio before the
request is sent.

### 4. Three different chunk configurations

The paper stated 500/50, `backend/rag_core.py` defaulted to 1000/200, and the
README suggested 800–1000/100–200.

**Fixed:** `config.ChunkConfig` is the only definition. The chunking ablation
measures the effect of varying it.

### 5. Similarity metric described backwards

The write-up described `IndexFlatL2` with a 0.55 distance ceiling. The code used
`IndexFlatIP` over normalised vectors with a 0.55 similarity floor — the same
number, the opposite comparison.

**Fixed:** documented explicitly in `index.py` and `retriever.py`, and the
manuscript text corrected to match.

### 6. Committed credentials

`backend/.env` contained a live `GROQ_API_KEY`. Removed; `.env` is gitignored;
the default pipeline requires no key. **Any key present in a fork of the earlier
history must be treated as compromised and rotated.**

## Structural changes

| v1 | v2 |
|---|---|
| `backend/rag_core.py` | `src/kannada_rag/{ingest,chunking,index,retriever,generation,pipeline}.py` |
| `backend/main.py` | `src/kannada_rag/api.py` |
| `rag.ipynb` | `python -m kannada_rag.cli` |
| Groq only | Ollama (default, local) + Groq (optional comparison) |
| No evaluation code | `src/kannada_rag/eval/` |
| No tests | `tests/` |

## The notebook

`rag.ipynb` was removed rather than updated. Its purpose — exploration and
one-off table generation — is exactly the workflow that produced tables which
disagreed with one another. Every experiment is now a CLI command whose output
is a logged, configuration-stamped artefact.
