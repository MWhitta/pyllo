"""Simple FAISS-backed vector store."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss  # type: ignore
import numpy as np

from .chunking import TextChunk


@dataclass
class VectorRecord:
    """Metadata stored alongside each embedding vector."""

    chunk_id: str
    source_id: str
    content: str
    metadata: dict


class VectorStore:
    """FAISS index wrapper with jsonl metadata persistence."""

    def __init__(self, index: faiss.Index, records: List[VectorRecord]):
        self.index = index
        self.records = records

    @classmethod
    def from_embeddings(cls, embeddings: np.ndarray, chunks: Iterable[TextChunk]) -> "VectorStore":
        embeddings = embeddings.copy()
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        records = [
            VectorRecord(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                content=chunk.content,
                metadata=chunk.metadata,
            )
            for chunk in chunks
        ]
        return cls(index=index, records=records)

    def save(self, path: Path) -> None:
        """Persist the FAISS index and metadata."""
        path.mkdir(parents=True, exist_ok=True)

        faiss_path = path / "index.faiss"
        meta_path = path / "records.jsonl"
        config_path = path / "config.json"

        faiss.write_index(self.index, str(faiss_path))

        with meta_path.open("w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")

        config = {"size": len(self.records), "dimension": self.index.d}
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        faiss_path = path / "index.faiss"
        meta_path = path / "records.jsonl"

        if not faiss_path.exists() or not meta_path.exists():
            raise FileNotFoundError(f"No vector store found under {path}.")

        index = faiss.read_index(str(faiss_path))
        records = []
        with meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                payload = json.loads(line)
                records.append(
                    VectorRecord(
                        chunk_id=payload["chunk_id"],
                        source_id=payload["source_id"],
                        content=payload["content"],
                        metadata=payload["metadata"],
                    )
                )
        return cls(index=index, records=records)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> List[List[Tuple[VectorRecord, float]]]:
        """Search the index and return records with similarity scores."""
        faiss.normalize_L2(query_embeddings)
        scores, indices = self.index.search(query_embeddings, top_k)
        output: List[List[Tuple[VectorRecord, float]]] = []
        for idx_list, score_list in zip(indices, scores):
            pair_list: List[Tuple[VectorRecord, float]] = []
            for idx, score in zip(idx_list, score_list):
                if idx == -1:
                    continue
                record = self.records[idx]
                pair_list.append((record, float(score)))
            output.append(pair_list)
        return output

