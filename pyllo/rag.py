"""High-level Retrieval-Augmented Generation interface."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .generator import ClayGenerator, GenerationResult
from .retriever import Retriever


@dataclass
class RAGAnswer:
    query: str
    answer: str
    context: list[str]


class ClayRAG:
    """Orchestrates retrieval and generation for the clay-science agent."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.settings.ensure_dirs()
        self.retriever = Retriever(settings)
        self.generator = ClayGenerator(settings.model)

    def answer(self, query: str) -> RAGAnswer:
        """Return an answer with supporting context for the provided query."""
        retrieved = self.retriever.retrieve(query)
        result: GenerationResult = self.generator.generate(query, retrieved)
        return RAGAnswer(query=query, answer=result.answer, context=result.context)

