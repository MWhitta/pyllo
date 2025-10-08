# Pyllo – Clay-Science RAG Agent

Pyllo ingests clay-science literature, builds a local vector store, and answers questions with citations.

## Quickstart
```bash
pip install -e .
pyllo ingest                                 # expects PDFs in data/literature/
export OPENAI_API_KEY="sk-..."               # or configure another backend
pyllo query "How do montmorillonite layers swell?"
```

## LLM Backends
- **Default (`litellm`)** – works with any provider litellm supports (OpenAI, Anthropic, Azure, local vLLM, etc.). Export the provider’s API key(s) before running queries.
- **CBORG (Berkeley Lab)** – OpenAI-compatible gateway for lab-hosted models:
```bash
export CBORG_API_KEY="cborg-..."
# Optional overrides (defaults already match these values):
export PYLLO_MODEL__PROVIDER=cborg
export PYLLO_MODEL__MODEL="gpt-5"
export PYLLO_MODEL__API_KEY_ENV=CBORG_API_KEY
export PYLLO_MODEL__API_BASE="https://api.cborg.lbl.gov"
export PYLLO_MODEL__MAX_TOKENS=128000
```
Discover current model identifiers:
```bash
pyllo cborg-models --show-details
```

## Helpful CLI Commands
- `pyllo ingest` – process PDFs and update `storage/vectorstore/`.
- `pyllo query "…"` – ask the clay expert, printing an answer plus retrieved context.
- `pyllo cborg-models --show-details` – list CBORG models and their API names.
- `pyllo minerals-download --mineral montmorillonite` – fetch Crossref manuscripts for minerals in `data/minerals/`.
- `pyllo structures-download --mineral Quartz --limit 1` – pull experimental (RRUFF) and simulated (Materials Project) CIF files into `data/structure/` (simulated files include the MP material id in the filename).

## Project Layout
- `data/literature/` – source PDFs (with optional `data/literature_metadata.jsonl`).
- `data/minerals/` – mineral datasets; downloads land in `data/minerals/manuscripts/`.
- `docs/` – design notes and user guide.
- `pyllo/` – Python package (ingestion, retrieval, CBORG utilities, CLI).
- `storage/` – generated vector store (created after ingestion).

## Ways to Extend
1. Expand `data/literature/` and metadata coverage for better recall.
2. Add evaluation notebooks to benchmark retrieval + generation quality.
3. Plug in additional ingestion transforms or alternative embedding models.
4. Explore local LLM endpoints via litellm for fully disconnected workflows.
