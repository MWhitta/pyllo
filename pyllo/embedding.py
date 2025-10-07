"""Embedding utilities for Pyllo."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EmbeddingConfig


@lru_cache(maxsize=1)
def load_model(config_tuple: tuple) -> SentenceTransformer:
    """Load and cache the embedding model."""
    model_name, device = config_tuple
    return SentenceTransformer(model_name, device=device or "cpu")


def embed_texts(texts: Iterable[str], config: EmbeddingConfig) -> np.ndarray:
    """Encode a list of texts into a numpy matrix."""
    model = load_model((config.model_name, config.device))
    embeddings = model.encode(list(texts), batch_size=config.batch_size, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype("float32")
