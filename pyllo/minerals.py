"""Utilities for gathering mineral reference manuscripts."""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import requests

DEFAULT_MINERAL_DATA_DIR = Path("data") / "minerals"
DEFAULT_OUTPUT_DIR = DEFAULT_MINERAL_DATA_DIR / "manuscripts"

USER_AGENT = os.environ.get(
    "PYLLO_MINERALS_USER_AGENT",
    "PylloMineralCollector/0.1 (+https://github.com/mwhittaker/pyllo)",
)


@dataclass
class Manuscript:
    """Represents a manuscript discovered for a mineral."""

    mineral: str
    title: str
    doi: Optional[str]
    source_url: str
    pdf_path: Optional[Path]
    published: Optional[str] = None


def slugify(value: str) -> str:
    """Return a filesystem-friendly slug for the provided string."""

    return "-".join(
        part.lower()
        for part in "".join(ch if ch.isalnum() else " " for ch in value).split()
        if part
    )


def read_mineral_names(mineral_dir: Path | None = None) -> List[str]:
    """Load mineral names from CSV files located in *mineral_dir*."""

    mineral_dir = mineral_dir or DEFAULT_MINERAL_DATA_DIR
    csv_paths = sorted(mineral_dir.glob("*.csv"))
    names: List[str] = []
    for csv_path in csv_paths:
        with csv_path.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                name = (row.get("Mineral Name") or "").strip()
                if not name:
                    continue
                names.append(name)
    # Preserve order while removing duplicates.
    seen = set()
    unique_names: List[str] = []
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        unique_names.append(name)
    return unique_names


def search_crossref(mineral: str, rows: int = 10) -> List[dict]:
    """Query Crossref for works mentioning the mineral in their titles."""

    url = "https://api.crossref.org/works"
    params = {
        "query.title": mineral,
        "filter": "type:journal-article",
        "rows": rows,
        "select": "DOI,title,URL,issued,link",
    }
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, params=params, headers=headers, timeout=30)
    response.raise_for_status()
    payload = response.json()
    return payload.get("message", {}).get("items", [])


def extract_pdf_link(item: dict) -> Optional[str]:
    """Return a candidate PDF link from a Crossref item, if available."""

    links = item.get("link") or []
    for link in links:
        if link.get("content-type") == "application/pdf" and link.get("URL"):
            return link["URL"]
    return None


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


class DownloadError(RuntimeError):
    """Raised when a manuscript download does not yield a valid PDF."""


def download_pdf(candidate_urls: Sequence[str], path: Path) -> None:
    """Attempt to download a PDF from the provided candidate URLs."""

    if path.exists():
        return

    attempts: List[str] = []
    for url in candidate_urls:
        if not url:
            continue
        headers = {"User-Agent": USER_AGENT, "Accept": "application/pdf"}
        try:
            with requests.get(
                url, headers=headers, timeout=60, stream=True, allow_redirects=True
            ) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").lower()
                if "pdf" not in content_type:
                    attempts.append(f"{url} -> {content_type or 'unknown'}")
                    continue

                first_bytes = b""
                with path.open("wb") as fp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        if len(first_bytes) < 8:
                            needed = 8 - len(first_bytes)
                            first_bytes += chunk[:needed]
                        fp.write(chunk)

                if b"%PDF" not in first_bytes:
                    path.unlink(missing_ok=True)
                    attempts.append(f"{url} -> missing %PDF signature")
                    continue

                return
        except requests.RequestException as exc:
            attempts.append(f"{url} -> {exc}")

    raise DownloadError("; ".join(attempts) or "No valid download URLs provided")


def collect_mineral_manuscripts(
    minerals: Sequence[str] | None = None,
    mineral_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    max_per_mineral: int = 3,
    crossref_rows: int = 12,
    sleep_seconds: float = 1.0,
    download: bool = True,
) -> List[Manuscript]:
    """Collect manuscripts for minerals and optionally download PDFs.

    Parameters
    ----------
    minerals:
        Specific mineral names to process. If omitted, names are taken from
        CSV exports in *mineral_dir*.
    mineral_dir:
        Directory containing mineral CSV exports (defaults to ``data/minerals``).
    output_dir:
        Directory where PDFs and metadata are stored (defaults to
        ``data/minerals/manuscripts``).
    max_per_mineral:
        Maximum number of manuscripts to keep per mineral.
    crossref_rows:
        Number of Crossref results fetched per mineral before filtering.
    sleep_seconds:
        Delay between Crossref requests to remain polite.
    download:
        If ``True`` download PDF files whenever a direct link is provided by
        Crossref. Set to ``False`` to only collect metadata.
    """

    mineral_dir = mineral_dir or DEFAULT_MINERAL_DATA_DIR
    output_dir = ensure_directory(output_dir or DEFAULT_OUTPUT_DIR)

    minerals = list(minerals or read_mineral_names(mineral_dir))
    collected: List[Manuscript] = []

    for mineral in minerals:
        try:
            results = search_crossref(mineral, rows=crossref_rows)
        except requests.RequestException as exc:
            print(f"[crossref] failed for {mineral}: {exc}")
            continue

        mineral_slug = slugify(mineral)
        mineral_dir_path = ensure_directory(output_dir / mineral_slug)

        hits = 0
        seen_keys = set()
        for item in results:
            title_list = item.get("title") or []
            title = title_list[0] if title_list else "Untitled"
            if mineral.lower() not in title.lower():
                continue
            doi = item.get("DOI")
            pdf_url = extract_pdf_link(item)
            doi_url = None
            if doi:
                doi_url = doi if doi.lower().startswith("http") else f"https://doi.org/{doi}"
            key = doi or title.lower()
            if key in seen_keys:
                continue
            seen_keys.add(key)

            source_url = item.get("URL") or ""
            pdf_path = None
            if download:
                pdf_filename = slugify(title) or (doi.replace("/", "-") if doi else "manuscript")
                pdf_path = mineral_dir_path / f"{pdf_filename}.pdf"
                candidates: List[str] = []
                if pdf_url:
                    candidates.append(pdf_url)
                if doi_url:
                    candidates.append(doi_url)
                if source_url:
                    candidates.append(source_url)
                try:
                    download_pdf(candidates, pdf_path)
                except (requests.RequestException, DownloadError) as exc:
                    print(f"[download] failed for {mineral}: {exc}")
                    pdf_path = None

            issued = item.get("issued", {}).get("date-parts", [])
            published = None
            if issued:
                parts = issued[0]
                published = "-".join(str(part) for part in parts)

            manuscript = Manuscript(
                mineral=mineral,
                title=title,
                doi=doi,
                source_url=source_url or pdf_url or (doi_url or ""),
                pdf_path=pdf_path,
                published=published,
            )
            collected.append(manuscript)
            hits += 1
            if hits >= max_per_mineral:
                break

        metadata_path = mineral_dir_path / "metadata.json"
        mineral_entries = []
        for m in collected:
            if m.mineral != mineral:
                continue
            entry = asdict(m)
            if entry.get("pdf_path"):
                entry["pdf_path"] = str(entry["pdf_path"])
            mineral_entries.append(entry)
        with metadata_path.open("w", encoding="utf-8") as fp:
            json.dump(mineral_entries, fp, indent=2)

        if sleep_seconds:
            time.sleep(sleep_seconds)

    return collected
