"""Top-level package for the Pyllo clay-science RAG toolkit."""

import os
import warnings

# Quiet common Hugging Face warnings and ensure consistent tokenizer behavior.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_RESUME", "1")

warnings.filterwarnings(
    "ignore",
    message=r"`resume_download` is deprecated and will be removed in version 1\.0\.0.*",
    category=FutureWarning,
    module="huggingface_hub.file_download",
)

from .cborg import cborg_models_as_csv, fetch_cborg_models
from .config import Settings
from .ingest import ingest_corpus
from .minerals import collect_mineral_manuscripts
from .rag import ClayRAG

__all__ = [
    "Settings",
    "ingest_corpus",
    "collect_mineral_manuscripts",
    "fetch_cborg_models",
    "cborg_models_as_csv",
    "ClayRAG",
]
