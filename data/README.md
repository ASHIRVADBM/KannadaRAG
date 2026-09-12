# Corpus

Place Kannada-language heritage PDFs here. `make index` ingests every `*.pdf` in
this directory.

## Rights

`HISTORICAL.pdf` is the corpus used in the accompanying study. Heritage
documents may carry copyright or institutional restrictions distinct from this
repository's MIT licence. Before redistributing any document placed here,
confirm you have the right to do so. The MIT licence covers the source code
only.

## What makes a good corpus document

- **Kannada text**, digital or scanned. Scanned pages are detected automatically
  (fewer than 50 characters of native text) and sent to Tesseract at 200 DPI.
- **Authoritative and curated.** The grounding guarantee is only as good as the
  corpus: the system will faithfully reproduce an error in a source document.
- **Page-structured.** Provenance is tracked per page, so citations are only as
  precise as the document's pagination.

## After adding documents

```bash
make index     # rebuilds; the config fingerprint invalidates the old index
```

Adding documents changes retrieval for every question, so re-run the evaluation
before comparing any number against a previously published one.
