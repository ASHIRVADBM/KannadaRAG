# Reproduction pipeline. Each target corresponds to a step described in the
# paper; running them in order regenerates every published number.

PY ?= python
MODELS ?= gemma3:4b llama3:8b mistral:7b phi3

.PHONY: help install index evaluate tables ablate ocr test lint serve clean reproduce

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:   ## Install the package with evaluation extras
	$(PY) -m pip install -e ".[all]"

index:     ## Ingest the corpus and build the FAISS index
	$(PY) -m kannada_rag.cli build

evaluate:  ## Run the benchmark for every model, with and without retrieval
	$(PY) -m kannada_rag.cli evaluate --models $(MODELS) --with-control --strict

tables:    ## Regenerate every table from results/records.jsonl
	$(PY) -m kannada_rag.cli tables

ablate:    ## Run every design-choice sweep
	$(PY) -m kannada_rag.cli ablate threshold
	$(PY) -m kannada_rag.cli ablate retrieval
	$(PY) -m kannada_rag.cli ablate chunking
	$(PY) -m kannada_rag.cli ablate encoder

ocr:       ## Measure OCR CER/WER against manual transcriptions
	$(PY) -m kannada_rag.cli ocr

test:      ## Run the test suite
	$(PY) -m pytest

lint:      ## Static checks
	ruff check src tests
	mypy src

serve:     ## Start the API and web interface at http://127.0.0.1:8000
	$(PY) -m kannada_rag.cli serve

reproduce: index evaluate tables ablate ## Full reproduction, start to finish
	@echo "All artefacts written to results/."

clean:
	rm -rf results/* vector_store/sweep_* vector_store/embed_* .pytest_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
