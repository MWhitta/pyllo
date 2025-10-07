"""LLM-powered answer generation for the clay-science assistant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from litellm import completion

from .config import ModelConfig
from .retriever import RetrievedChunk


SYSTEM_PROMPT = (
    "You are Pyllo, an AI clay-science expert. "
    "Answer questions about clay minerals, geochemistry, and industrial applications with precision. "
    "Ground every statement in the provided context snippets and include citations in [AuthorYear] format. "
    "If the context is insufficient, say so and suggest what information would help."
)


def build_context(chunks: Iterable[RetrievedChunk]) -> List[str]:
    """Format retrieved chunks into context strings with citations."""
    formatted = []
    for item in chunks:
        meta = item.record.metadata
        citation = meta.get("citation") or meta.get("source_id") or item.record.source_id
        formatted.append(
            f"[{citation}] Score={item.score:.3f}\n{item.record.content.strip()}"
        )
    return formatted


@dataclass
class GenerationResult:
    answer: str
    context: List[str]


class ClayGenerator:
    """LLM client used to synthesize answers from retrieved chunks."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> GenerationResult:
        """Generate an answer conditioned on retrieved context."""
        context_blocks = build_context(chunks)
        context_text = "\n\n".join(context_blocks) if context_blocks else "No supporting context available."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\n"
                    "Context:\n"
                    f"{context_text}\n\n"
                    "Compose a detailed answer as Pyllo. Cite sources using [AuthorYear] tags."
                ),
            },
        ]

        try:
            response = completion(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            answer = response["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # pragma: no cover - defensive logging
            answer = (
                "Generation failed because the LLM call raised an error. "
                f"Details: {exc}. Verify API credentials and model availability."
            )

        return GenerationResult(answer=answer, context=context_blocks)
