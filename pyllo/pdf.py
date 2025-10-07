"""PDF processing helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import fitz  # type: ignore


@dataclass
class PageContent:
    """Container for page-level text and metadata."""

    page_number: int
    text: str


class PDFExtractor:
    """Basic PDF text extractor using PyMuPDF."""

    def __init__(self, *, include_headers: bool = True) -> None:
        self.include_headers = include_headers

    def extract(self, path: Path) -> Iterable[PageContent]:
        """Yield cleaned text for each page."""
        with fitz.open(path) as doc:
            for index, page in enumerate(doc):
                text = page.get_text("text")
                yield PageContent(page_number=index + 1, text=text)


def extract_full_text(path: Path) -> Tuple[str, int]:
    """Convenience helper to load a PDF to a single text string and page count."""
    extractor = PDFExtractor()
    pages = list(extractor.extract(path))
    combined = "\n\n".join(page.text for page in pages)
    return combined, len(pages)

