# Clay-Science RAG System Plan

## Goals
- Assemble an extensible knowledge base that covers peer-reviewed clay-science literature.
- Deliver a retrieval-augmented generation (RAG) assistant that can answer clay-focused questions with strong sourcing.
- Keep ingestion and retrieval modular so we can iterate on models, storage, and data curation independently.

## Data & Collection Strategy
- **Scope**: start with review articles and textbooks to give broad coverage, then layer in specialized topics (structure, sorption, geochemistry, industrial applications).
- **Sources**: MDPI, Springer open access, arXiv, USGS reports, clay conference proceedings. Capture metadata (title, authors, DOI, year, venue, keywords).
- **Acquisition**:
  - Maintain raw PDFs under `data/literature/raw/`.
  - Track provenance and licensing in `data/literature_metadata.jsonl`.
  - Provide a `scripts/download.py` helper that can ingest a DOI list or local files.
- **Versioning**: each ingestion run logs SHA256 hashes of the PDF to detect updates and avoid duplicates.

## Ingestion Pipeline
1. **Parse**: convert PDF to structured text using `pymupdf` (fast and handles scientific PDFs). Store intermediate `.json` with page-level text + layout metadata.
2. **Clean**: normalize whitespace, remove references and figure captions when requested, preserve equations as inline text markers.
3. **Chunk**: split into semantic units (~500 tokens) using heading cues; fallback to recursive character splitting with overlap.
4. **Metadata enrichment**: attach content type (abstract, methods, results), citation, keywords, and any clay-mineral tags extracted via keyword spotting.
5. **Embed**: encode chunks with `sentence-transformers` (`all-MiniLM-L6-v2`) for general semantic coverage. Allow swapping models via config.
6. **Store**: persist vectors in a FAISS index on disk (`storage/index.faiss`) plus a sidecar parquet/jsonl for metadata. Keep ingestion idempotent by reusing existing ids when hashes match.

## Retrieval & Generation
- **Retriever**: FAISS similarity search with metadata filters (e.g., clay type, publication year).
- **Reranker (optional)**: integrate a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) when we need higher precision.
- **LLM**: default to OpenAI GPT-4o via `litellm` wrapper; allow environment variable `PYLLO_LLM_MODEL` to target alternates (Claude, local vLLM).
- **Prompting**: build a structured prompt template that:
  - Summarizes the question and sets clay-science expert persona.
  - Injects citations using `[AuthorYear]` format referencing chunk metadata.
  - Encourages hedging when retrieval confidence is low.

## Evaluation
- Spot-check answers against known results from foundational clay texts.
- Build a small set of clay-specific QA pairs to run regression tests.
- Track retrieval diagnostics (recall@k, coverage per document) via a nightly notebook.

## Deliverables (near term)
- `pyproject.toml` defining dependencies.
- `pyllo/` package with:
  - `ingest.py` for pipeline orchestration.
  - `retriever.py` and `generator.py`.
  - `cli.py` offering `ingest`, `query`, `chat` commands.
- `notebooks/Exploratory.ipynb` template for exploratory retrieval.
- Documentation (README updates + `docs/usage.md`) covering setup, ingestion, querying, and extension points.

## Roadmap
1. Stand up minimal pipeline over existing review PDFs.
2. Expand the literature library with domain coverage tracking dashboard.
3. Add confidence calibration (score thresholds, abstain logic).
4. Integrate structured knowledge (e.g., mineral property tables) to support hybrid retrieval.
5. Eventually incorporate agentic workflows (e.g., multi-hop reasoning, tool use for property lookup).
