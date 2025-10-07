"""Configuration helpers for the Pyllo RAG toolkit."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """LLM model configuration."""

    provider: str = Field(default="litellm", description="Provider identifier handled by litellm.")
    model: str = Field(default="gpt-4o", description="Model name to use for generation.")
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, gt=1)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    batch_size: int = Field(default=16, gt=0)
    device: Optional[str] = Field(default=None, description="Torch device override, e.g. 'cpu' or 'cuda'.")

    model_config = ConfigDict(protected_namespaces=())


class RetrieverConfig(BaseModel):
    """Retriever behavior configuration."""

    top_k: int = Field(default=5, gt=0)
    reranker_model: Optional[str] = Field(
        default=None,
        description="Optional cross encoder model for reranking; if unset retrieval uses only embeddings.",
    )


DEFAULT_LITERATURE_DIR = Path("data") / "literature"
DEFAULT_METADATA_PATH = Path("data") / "literature_metadata.jsonl"


class Settings(BaseSettings):
    """Application settings pulled from environment variables."""

    data_dir: Path = Field(default_factory=lambda: Path("storage"))
    corpus_dirs: List[Path] = Field(default_factory=lambda: [DEFAULT_LITERATURE_DIR])
    metadata_path: Path = Field(default=DEFAULT_METADATA_PATH)
    cache_dir: Optional[Path] = None

    model: ModelConfig = Field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)

    class Config:
        env_prefix = "PYLLO_"
        env_nested_delimiter = "__"

    def ensure_dirs(self) -> None:
        """Create directories required by the system if they do not exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for corpus_dir in self.corpus_dirs:
            corpus_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
