"""Retriever component built on top of the vector store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .config import EmbeddingConfig, RetrieverConfig, Settings
from .embedding import embed_texts
from .vectorstore import VectorRecord, VectorStore


@dataclass
class RetrievedChunk:
    record: VectorRecord
    score: float


class Retriever:
    """Lightweight wrapper around the vector store for semantic retrieval."""

    def __init__(
        self,
        settings: Settings,
        *,
        embedding_config: EmbeddingConfig | None = None,
        retriever_config: RetrieverConfig | None = None,
        store_path: Path | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_config = embedding_config or settings.embedding
        self.retriever_config = retriever_config or settings.retriever
        self.store_path = store_path or settings.data_dir / "vectorstore"
        self.store = VectorStore.load(self.store_path)

    def retrieve(self, query: str, *, top_k: int | None = None) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for the provided query string."""
        top_k = top_k or self.retriever_config.top_k
        query_emb = embed_texts([query], self.embedding_config)
        results = self.store.search(query_emb, top_k=top_k)[0]
        return [RetrievedChunk(record=record, score=score) for record, score in results]

