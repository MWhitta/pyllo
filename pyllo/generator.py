"""LLM-powered answer generation for the clay-science assistant."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI
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


def _normalize_message_content(message: dict) -> str:
    """Extract textual content from an OpenAI-style chat message."""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                item = item.strip()
                if item:
                    parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            if "text" in item and isinstance(item["text"], str):
                text = item["text"].strip()
                if text:
                    parts.append(text)
                    continue
            if "value" in item and isinstance(item["value"], str):
                text = item["value"].strip()
                if text:
                    parts.append(text)
                    continue
            if item.get("type") in {"text", "output_text"} and isinstance(item.get("text"), str):
                text = item["text"].strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""

def _sanitize_for_api(text: str) -> str:
    """Return ASCII-safe text to avoid downstream encoding errors."""
    if not isinstance(text, str):
        text = str(text)
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        return text.encode("ascii", errors="replace").decode("ascii")


class ClayGenerator:
    """LLM client used to synthesize answers from retrieved chunks."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.provider = (config.provider or "litellm").lower()

    def _cborg_base_url(self) -> str:
        if self.config.api_base:
            return self.config.api_base.rstrip("/")
        env_base = os.getenv("CBORG_API_BASE")
        if env_base:
            return env_base.rstrip("/")
        return "https://api.cborg.lbl.gov"

    def _cborg_api_key(self) -> str:
        env_name = self.config.api_key_env or "CBORG_API_KEY"
        api_key = os.getenv(env_name)
        if not api_key:
            raise RuntimeError(
                f"Missing API key for CBORG provider. Set the environment variable '{env_name}'."
            )
        return api_key

    _cborg_client: Optional[OpenAI] = None

    def _get_cborg_client(self) -> OpenAI:
        base_url = self._cborg_base_url()
        api_key = self._cborg_api_key()
        if not self._cborg_client:
            self._cborg_client = OpenAI(api_key=api_key, base_url=base_url)
        return self._cborg_client

    def _extract_answer(self, data: dict) -> str:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Unexpected response payload without choices: {data}")
        message = choices[0].get("message") or {}
        text = _normalize_message_content(message)
        if text:
            return text
        # Legacy fallback
        if "text" in choices[0] and isinstance(choices[0]["text"], str):
            text_value = choices[0]["text"].strip()
            if text_value:
                return text_value
        raise RuntimeError(f"No textual content found in response: {data}")

    def _invoke_cborg(self, messages: List[dict]) -> dict:
        client = self._get_cborg_client()
        response = client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.model_dump()

    def generate(self, query: str, chunks: List[RetrievedChunk]) -> GenerationResult:
        """Generate an answer conditioned on retrieved context."""
        context_blocks = build_context(chunks)
        context_text = "\n\n".join(context_blocks) if context_blocks else "No supporting context available."
        safe_query = _sanitize_for_api(query)
        safe_context = _sanitize_for_api(context_text)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{safe_query}\n\n"
                    "Context:\n"
                    f"{safe_context}\n\n"
                    "Compose a detailed answer as Pyllo. Cite sources using [AuthorYear] tags."
                ),
            },
        ]

        try:
            if self.provider == "cborg":
                data = self._invoke_cborg(messages)
                answer = self._extract_answer(data)
            else:
                response = completion(
                    model=self.config.model,
                    messages=messages,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                answer = self._extract_answer(response)
        except Exception as exc:  # pragma: no cover - defensive logging
            answer = (
                "Generation failed because the LLM call raised an error. "
                f"Details: {exc}. Verify API credentials, base URL, and model availability."
            )

        return GenerationResult(answer=answer, context=context_blocks)
