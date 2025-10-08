"""Corpus ingestion entrypoints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from rich.console import Console
from rich.progress import track

from .chunking import TextChunk, chunk_text
from .config import Settings
from .embedding import embed_texts
from .pdf import extract_full_text
from .vectorstore import VectorStore

console = Console()


@dataclass
class DocumentMetadata:
    source_id: str
    citation: str
    path: Path
    sha256: str
    page_count: int
    extra: Dict[str, str]


def sha256_file(path: Path) -> str:
    """Return SHA256 hash for a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_metadata_map(metadata_path: Path) -> Dict[str, dict]:
    """Load optional metadata definitions keyed by filename stem."""
    if not metadata_path.exists():
        return {}
    mapping: Dict[str, dict] = {}
    with metadata_path.open("r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            key = entry.get("source_id") or entry.get("slug") or entry.get("filename")
            if key:
                mapping[str(key)] = entry
    return mapping


def discover_pdfs(dirs: Iterable[Path]) -> List[Path]:
    """Find all PDF files under the provided directories."""
    pdfs: List[Path] = []
    for dir_path in dirs:
        if not dir_path.exists():
            continue
        for path in dir_path.rglob("*.pdf"):
            pdfs.append(path)
    return sorted({p.resolve() for p in pdfs})


def build_document_metadata(
    path: Path, metadata_map: Dict[str, dict], *, page_count: int
) -> DocumentMetadata:
    """Assemble metadata for a single PDF."""
    sha = sha256_file(path)
    key = path.stem
    entry = metadata_map.get(key, {})

    citation = entry.get("citation") or entry.get("title") or path.stem
    source_id = entry.get("source_id") or key

    extra = dict(entry)
    extra.update(
        {
            "sha256": sha,
            "page_count": page_count,
            "filename": path.name,
        }
    )
    return DocumentMetadata(
        source_id=source_id,
        citation=citation,
        path=path,
        sha256=sha,
        page_count=page_count,
        extra=extra,
    )


def ingest_corpus(settings: Settings | None = None) -> Path:
    """Ingest PDFs into an on-disk vector store and return its path."""
    settings = settings or Settings()
    settings.ensure_dirs()

    metadata_map = load_metadata_map(settings.metadata_path)
    pdf_paths = discover_pdfs(settings.corpus_dirs)

    if not pdf_paths:
        console.print(
            "[yellow]No PDF files found. Place documents under the literature "
            "directory (data/literature by default) and retry."
        )
        raise FileNotFoundError("No literature PDFs discovered for ingestion.")

    chunks: List[TextChunk] = []
    raw_texts: List[str] = []

    console.print(f"[cyan]Found {len(pdf_paths)} PDF files. Beginning ingestion...")

    for path in track(pdf_paths, description="Processing PDFs"):
        try:
            text, page_count = extract_full_text(path)
        except FileNotFoundError:
            console.print(f"[red]Skipping missing file: {path}[/red]")
            continue
        except Exception as exc:
            console.print(f"[red]Failed to parse {path}: {exc}[/red]")
            continue
        doc_meta = build_document_metadata(path, metadata_map, page_count=page_count)

        base_meta = {
            "citation": doc_meta.citation,
            "source_id": doc_meta.source_id,
            "sha256": doc_meta.sha256,
            "filename": doc_meta.path.name,
        }
        base_meta.update(doc_meta.extra)

        doc_chunks = list(
            chunk_text(
                text,
                source_id=doc_meta.source_id,
                base_metadata=base_meta,
            )
        )
        if not doc_chunks:
            console.print(f"[yellow]Skipping empty document: {path}")
            continue

        chunks.extend(doc_chunks)
        raw_texts.extend(chunk.content for chunk in doc_chunks)

    console.print(f"[cyan]Creating embeddings for {len(chunks)} chunks...")
    embeddings = embed_texts(raw_texts, settings.embedding)

    store = VectorStore.from_embeddings(embeddings, chunks)
    store_path = settings.data_dir / "vectorstore"
    store.save(store_path)
    console.print(f"[green]Vector store saved to {store_path}")

    return store_path
