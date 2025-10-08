"""Utilities for turning raw document text into retrieval chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional

HEADING_PATTERN = re.compile(r"^\s*((\d+(\.\d+)*)|[A-Z][A-Z\s]{2,})[\.\)]?\s")


@dataclass
class TextChunk:
    """Represents a chunk of processed text ready for embedding."""

    content: str
    source_id: str
    chunk_id: str
    metadata: dict


def clean_text(text: str) -> str:
    """Lightly normalize whitespace and remove spurious artefacts."""
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_by_headings(text: str) -> List[str]:
    """Split on headings when possible to keep semantic context together."""
    lines = text.splitlines()
    sections: List[str] = []
    buffer: List[str] = []

    def flush():
        if buffer:
            sections.append("\n".join(buffer).strip())
            buffer.clear()

    for line in lines:
        if HEADING_PATTERN.match(line) and buffer:
            flush()
        buffer.append(line)
    flush()
    return [s for s in sections if s]


def chunk_text(
    text: str,
    source_id: str,
    base_metadata: Optional[dict] = None,
    target_tokens: int = 500,
    overlap_tokens: int = 75,
) -> Iterable[TextChunk]:
    """Split text into overlapping chunks ready for embedding."""
    base_metadata = base_metadata or {}
    sections = split_by_headings(clean_text(text))
    if not sections:
        sections = [clean_text(text)]

    chunk_idx = 0
    for section in sections:
        words = section.split()
        start = 0
        stride = max(target_tokens - overlap_tokens, 1)
        while start < len(words):
            end = min(start + target_tokens, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            if not chunk_text.strip():
                break

            metadata = dict(base_metadata)
            metadata["section_start_word"] = start
            metadata["section_end_word"] = end
            metadata.setdefault("section_title", words[0] if words else "")

            yield TextChunk(
                content=chunk_text,
                source_id=source_id,
                chunk_id=f"{source_id}::chunk-{chunk_idx}",
                metadata=metadata,
            )
            chunk_idx += 1
            start += stride
