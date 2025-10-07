# Pyllo - Clay-Science RAG Agent

Pyllo is a Retrieval Augmented Generation (RAG) toolkit for building a world-class clay science assistant. It ingests clay-mineral literature (PDFs), embeds the content into a vector store, and serves expert answers grounded in citations.

## Quickstart

```bash
pip install -e .
pyllo ingest
pyllo query "How do montmorillonite swelling properties vary with cation exchange?"
```

Set the `OPENAI_API_KEY` (or any provider supported by `litellm`) before running queries.

## Project Layout

- `data/literature/` - place raw PDFs (nested folders allowed).
- `data/literature_metadata.jsonl` - optional citation/keyword metadata.
- `docs/rag_plan.md` - system design and roadmap.
- `pyllo/` - ingestion pipeline, vector store, retriever, generator, CLI.
- `storage/` - generated FAISS vector store (created after ingestion).

## Next Steps

1. Expand `data/literature/` with additional clay-science sources.
2. Populate `data/literature_metadata.jsonl` with DOI, citation, and keyword metadata.
3. Implement evaluation notebooks for answer quality and retrieval diagnostics.

See `docs/rag_plan.md` for more details.
