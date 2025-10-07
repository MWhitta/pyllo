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


def _collect_text_fragments(value: object, *, allow_reasoning_text: bool = False) -> List[str]:
    """Recursively gather textual fragments from OpenAI response structures."""

    fragments: List[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        cleaned = text.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            fragments.append(cleaned)

    def visit(obj: object, allow_free_text: bool) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            add(obj)
            return
        if isinstance(obj, dict):
            obj_type = obj.get("type")
            candidate_keys = ("output_text", "final_answer", "answer")
            found_candidate = False
            for key in candidate_keys:
                if key in obj:
                    found_candidate = True
                    visit(obj[key], allow_free_text=False)
            if found_candidate:
                return

            text_value = obj.get("text")
            if isinstance(text_value, (str, list, dict)):
                if allow_free_text or obj_type in {"output_text", "final_answer", "answer", "text"}:
                    visit(text_value, allow_free_text=False)

            value_field = obj.get("value")
            if isinstance(value_field, (str, list, dict)):
                if allow_free_text or obj_type in {"output_text", "final_answer", "answer", "text"}:
                    visit(value_field, allow_free_text=False)

            nested_keys = (
                "steps",
                "segments",
                "content",
                "parts",
                "messages",
                "items",
                "choices",
                "reasoning",
            )
            for key in nested_keys:
                if key in obj:
                    visit(obj[key], allow_free_text=True)
            return
        if isinstance(obj, list):
            for item in obj:
                visit(item, allow_free_text=allow_free_text)

    visit(value, allow_free_text=allow_reasoning_text)
    return fragments


def _extract_text_from_reasoning(reasoning: object) -> str:
    """Pull the final textual answer out of OpenAI reasoning payloads."""
    fragments = _collect_text_fragments(reasoning or {}, allow_reasoning_text=False)
    if not fragments:
        fragments = _collect_text_fragments(reasoning or {}, allow_reasoning_text=True)
    return "\n".join(fragments).strip()


def _normalize_message_content(message: dict) -> str:
    """Extract textual content from an OpenAI-style chat message."""
    if not isinstance(message, dict):
        return ""

    for key in ("output_text", "final_answer", "answer"):
        fragments = _collect_text_fragments(message.get(key), allow_reasoning_text=False)
        if fragments:
            return "\n".join(fragments).strip()

    reasoning_text = _extract_text_from_reasoning(message.get("reasoning"))
    if reasoning_text:
        return reasoning_text

    content_fragments = _collect_text_fragments(message.get("content"), allow_reasoning_text=True)
    if content_fragments:
        return "\n".join(content_fragments).strip()
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
        for key in ("output_text", "text", "final_answer", "answer"):
            fragments = _collect_text_fragments(choices[0].get(key), allow_reasoning_text=True)
            if fragments:
                return "\n".join(fragments).strip()
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
