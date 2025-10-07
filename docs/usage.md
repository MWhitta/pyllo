# Pyllo Usage Guide

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 2. Prepare Literature Library

- Drop PDFs into `data/literature/` (nested folders allowed).
- (Optional) Add metadata entries to `data/literature_metadata.jsonl`:

```jsonl
{"source_id": "klopprogge2022-mdpi", "slug": "Klopprogge2022 MDPI", "title": "Recent Advances in Clay Science", "year": 2022, "citation": "Klopprogge 2022 MDPI Review"}
```

## 3. Build the Vector Store

```bash
pyllo ingest
```

Artifacts are written to `storage/vectorstore/`.

## 4. Ask Questions

```bash
export OPENAI_API_KEY=...
pyllo query "Summarize smectite cation exchange trends."
```

Add `--top-k` or `--no-show-context` flags as needed.

## 5. Extend

- Swap embedding or LLM models via environment variables (e.g., `PYLLO_MODEL__MODEL=claude-3-sonnet`).
- Integrate new ingestion transforms inside `pyllo/ingest.py` as additional preprocessing steps.
- Track ingestion runs by storing logs in a versioned directory under `storage/`.

Refer to `docs/rag_plan.md` for roadmap and architectural details.
