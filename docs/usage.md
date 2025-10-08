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

## 6. Use CBORG Models

CBORG exposes an OpenAI-compatible API. Configure Pyllo with the CBORG provider before querying:

```bash
export CBORG_API_KEY="cborg-..."
export PYLLO_MODEL__PROVIDER=cborg
export PYLLO_MODEL__MODEL="aws/llama-3.1-405b"
export PYLLO_MODEL__API_KEY_ENV=CBORG_API_KEY
export PYLLO_MODEL__API_BASE="https://api.cborg.lbl.gov"   # append /openai/v1 if your deployment requires it
```

Replace the model name with any identifier from https://cborg.lbl.gov/models/.

List available identifiers locally:

```bash
pyllo cborg-models --show-details
```

The `API Name(s)` column in the output contains the string to use with `PYLLO_MODEL__MODEL`.

## 7. Collect Mineral Manuscripts

Use Crossref to gather referenced manuscripts for IMA-approved minerals listed in `data/minerals/`:

```bash
pyllo minerals-download --mineral montmorillonite --mineral kaolinite --max-per-mineral 2
```

Downloads (when permitted) land in `data/minerals/manuscripts/`, alongside JSON metadata for later ingestion.
Set `PYLLO_MINERALS_USER_AGENT` to include your contact details for polite Crossref access.

## 8. Collect Crystal Structures

- Ensure a RRUFF mineral export (e.g., `data/minerals/rag-minerals-rruff-export-*.csv`) exists.
- Download experimental and simulated CIF files into `data/structure/experimental/` and `data/structure/simulated/`:

```bash
pyllo structures-download --mineral Quartz --limit 1
```

- Provide a Materials Project API key via `--materials-api-key` or the `MAPI_KEY`/`MATERIALS_PROJECT_API_KEY` environment variables to enable simulated structures. Install `pymatgen` (`pip install pymatgen`) if you have not already. Simulated CIFs are saved as `data/structure/simulated/mp-<mineral>-<mpid>.cif` so polymorphs are distinguished.
