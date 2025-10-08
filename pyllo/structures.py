"""Utilities for downloading crystal structures for minerals."""

from __future__ import annotations

import csv
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from rich.console import Console

RRUFF_SEARCH_URL = "https://rruff.geo.arizona.edu/AMS/result.php"
RRUFF_BASE_URL = "https://rruff.geo.arizona.edu"
MATERIALS_SUMMARY_URL = "https://api.materialsproject.org/materials/summary/"


@dataclass(frozen=True)
class MineralRecord:
    """Basic mineral metadata parsed from the RRUFF CSV export."""

    name: str
    formula: Optional[str]
    elements: Sequence[str]


@dataclass
class DownloadResult:
    """Outcome from attempting to download a structure file."""

    mineral: MineralRecord
    source: str
    status: str
    message: str
    path: Optional[Path] = None


class StructureDownloaderError(RuntimeError):
    """Raised when structural downloads cannot proceed."""


def slugify(text: str) -> str:
    sanitized = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return sanitized.strip("-")


def ensure_structure_dirs(base_dir: Path) -> tuple[Path, Path]:
    experimental_dir = base_dir / "structure" / "experimental"
    simulated_dir = base_dir / "structure" / "simulated"
    experimental_dir.mkdir(parents=True, exist_ok=True)
    simulated_dir.mkdir(parents=True, exist_ok=True)
    return experimental_dir, simulated_dir


def read_mineral_records(
    csv_path: Path, restrict_to: Optional[Sequence[str]] = None, limit: Optional[int] = None
) -> List[MineralRecord]:
    restrict_normalized = {name.strip().lower() for name in restrict_to or []}
    records: List[MineralRecord] = []

    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row.get("Mineral Name", "").strip()
            if not name:
                continue
            if restrict_normalized and name.lower() not in restrict_normalized:
                continue
            formula = row.get("IMA Chemistry (plain)") or row.get("RRUFF Chemistry (plain)") or None
            elements_raw = row.get("Chemistry Elements") or ""
            elements = tuple(
                sorted({item for item in elements_raw.replace(",", " ").split() if item})
            )
            records.append(MineralRecord(name=name, formula=formula, elements=elements))
            if limit and len(records) >= limit:
                break
    return records


def normalize_formula(formula: str) -> Optional[str]:
    if not formula:
        return None

    sanitized = formula.replace(" ", "")
    sanitized = sanitized.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
    sanitized = sanitized.replace("^", "")
    sanitized = re.sub(r"(\d)[+-]", r"\1", sanitized)
    sanitized = re.sub(r"[+-]", "", sanitized)

    parts = re.split(r"[·•∙]", sanitized)

    try:
        from pymatgen.core import Composition
    except ImportError as exc:
        raise StructureDownloaderError(
            "pymatgen is required to normalize mineral formulas. Install pymatgen to continue."
        ) from exc

    total = None
    for part in parts:
        chunk = part.strip()
        if not chunk:
            continue
        try:
            comp = Composition(chunk)
        except Exception:
            return None
        if total is None:
            total = comp
        else:
            total += comp

    if total is None:
        return None

    return total.reduced_formula


def download_rruff_cif(
    mineral: MineralRecord,
    output_dir: Path,
    *,
    session: Optional[requests.Session] = None,
    sleep_seconds: float = 0.8,
) -> DownloadResult:
    sess = session or requests.Session()
    payload = {
        "Mineral": mineral.name,
        "Author": "",
        "Periodic": "",
        "CellParam": "",
        "diff": "",
        "Key": "",
        "logic": "AND",
        "Viewing": "cif",
        "Download": "cif",
    }

    try:
        response = sess.post(RRUFF_SEARCH_URL, data=payload, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        return DownloadResult(
            mineral=mineral,
            source="rruff",
            status="error",
            message=f"RRUFF request failed: {exc}",
        )

    soup = BeautifulSoup(response.text, "html.parser")
    cif_links: List[str] = []
    for anchor in soup.find_all("a"):
        href = anchor.get("href")
        if not href:
            continue
        if "down=cif" in href:
            cif_links.append(urljoin(RRUFF_BASE_URL, href))

    if not cif_links:
        return DownloadResult(
            mineral=mineral,
            source="rruff",
            status="missing",
            message="No CIF links found in AMCSD search results.",
        )

    slug = slugify(mineral.name)
    target_path = output_dir / f"rruff-{slug}.cif"
    if target_path.exists():
        return DownloadResult(
            mineral=mineral,
            source="rruff",
            status="exists",
            message="CIF already downloaded.",
            path=target_path,
        )

    cif_url = cif_links[0]

    try:
        cif_response = sess.get(cif_url, timeout=30)
        cif_response.raise_for_status()
    except requests.RequestException as exc:
        return DownloadResult(
            mineral=mineral,
            source="rruff",
            status="error",
            message=f"Failed to fetch CIF: {exc}",
        )

    target_path.write_bytes(cif_response.content)
    if sleep_seconds:
        time.sleep(sleep_seconds)

    return DownloadResult(
        mineral=mineral,
        source="rruff",
        status="downloaded",
        message=f"Saved CIF from {cif_url}",
        path=target_path,
    )


def download_materials_project_cif(
    mineral: MineralRecord,
    output_dir: Path,
    *,
    api_key: Optional[str] = None,
    session: Optional[requests.Session] = None,
    sleep_seconds: float = 0.5,
) -> DownloadResult:
    key = api_key or os.environ.get("MAPI_KEY") or os.environ.get("MATERIALS_PROJECT_API_KEY")
    if not key:
        return DownloadResult(
            mineral=mineral,
            source="materials_project",
            status="skipped",
            message="Materials Project API key not provided.",
        )

    if not mineral.formula:
        return DownloadResult(
            mineral=mineral,
            source="materials_project",
            status="skipped",
            message="Mineral entry lacks a chemical formula.",
        )

    formula = normalize_formula(mineral.formula)
    if not formula:
        return DownloadResult(
            mineral=mineral,
            source="materials_project",
            status="skipped",
            message="Unable to normalize mineral formula for Materials Project search.",
        )

    sess = session or requests.Session()
    headers = {"X-API-KEY": key}
    params = {
        "formula": formula,
        "_fields": "material_id,formula_pretty,structure,energy_per_atom",
        "_limit": 10,
        "_sort_fields": "energy_per_atom",
    }

    try:
        summary_response = sess.get(
            MATERIALS_SUMMARY_URL, params=params, headers=headers, timeout=30
        )
    except requests.RequestException as exc:
        return DownloadResult(
            mineral=mineral,
            source="materials_project",
            status="error",
            message=f"Materials Project summary request failed: {exc}",
        )

    if summary_response.status_code == 401:
        return DownloadResult(
            mineral=mineral,
            source="materials_project",
            status="error",
            message="Materials Project authentication failed. Check the API key.",
        )

    if not summary_response.ok:
        detail = ""
        try:
            detail = summary_response.json()
        except Exception:
            detail = summary_response.text
        return DownloadResult(
            mineral=mineral,
            source="materials_project",
            status="error",
            message=f"Materials Project summary error ({summary_response.status_code}): {detail}",
        )

    data = summary_response.json()
    entries = data.get("data") if isinstance(data, dict) else None
    if not entries:
        return DownloadResult(
            mineral=mineral,
            source="materials_project",
            status="missing",
            message=f"No Materials Project entries found for formula {formula}.",
        )

    selected_entry = None
    for candidate in entries:
        if candidate.get("structure") and candidate.get("material_id"):
            selected_entry = candidate
            break

    if not selected_entry:
        return DownloadResult(
            mineral=mineral,
            source="materials_project",
            status="error",
            message="Materials Project response lacked usable structure entries.",
        )

    material_id = selected_entry["material_id"]
    structure_payload = selected_entry.get("structure")
    if not structure_payload:
        return DownloadResult(
            mineral=mineral,
            source="materials_project",
            status="error",
            message=(
                "Materials Project response lacked structure data. Try widening the "
                "query or checking API permissions."
            ),
        )

    energy_pa = selected_entry.get("energy_per_atom")

    slug = slugify(mineral.name)
    target_path = output_dir / f"mp-{slug}-{material_id}.cif"
    if target_path.exists():
        return DownloadResult(
            mineral=mineral,
            source="materials_project",
            status="exists",
            message="CIF already downloaded.",
            path=target_path,
        )

    try:
        from pymatgen.core import Structure
    except ImportError as exc:
        raise StructureDownloaderError(
            "pymatgen is required to serialize Materials Project structures. "
            "Install pymatgen to continue."
        ) from exc

    try:
        structure = Structure.from_dict(structure_payload)
    except Exception as exc:
        return DownloadResult(
            mineral=mineral,
            source="materials_project",
            status="error",
            message=f"Failed to parse Materials Project structure payload: {exc}",
        )

    cif_data = structure.to(fmt="cif")
    target_path.write_text(cif_data)
    if sleep_seconds:
        time.sleep(sleep_seconds)

    message = f"Saved CIF for {material_id}"
    if energy_pa is not None:
        message += f" (energy_per_atom={energy_pa:.6f} eV)"

    return DownloadResult(
        mineral=mineral,
        source="materials_project",
        status="downloaded",
        message=message,
        path=target_path,
    )


def gather_structures(
    *,
    csv_path: Path,
    base_dir: Path,
    minerals: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    include_experimental: bool = True,
    include_simulated: bool = True,
    api_key: Optional[str] = None,
    sleep_seconds: float = 0.5,
    console: Optional[Console] = None,
) -> List[DownloadResult]:
    console = console or Console()
    experimental_dir, simulated_dir = ensure_structure_dirs(base_dir)

    records = read_mineral_records(csv_path, restrict_to=minerals, limit=limit)
    if not records:
        raise StructureDownloaderError("No minerals matched the provided filters.")

    results: List[DownloadResult] = []
    session = requests.Session()

    for mineral in records:
        if include_experimental:
            result = download_rruff_cif(
                mineral, experimental_dir, session=session, sleep_seconds=sleep_seconds
            )
            results.append(result)
            console.log(f"[cyan]RRUFF[/cyan] {mineral.name}: {result.status} - {result.message}")

        if include_simulated:
            result = download_materials_project_cif(
                mineral,
                simulated_dir,
                api_key=api_key,
                session=session,
                sleep_seconds=sleep_seconds,
            )
            results.append(result)
            console.log(
                f"[magenta]Materials Project[/magenta] {mineral.name}: "
                f"{result.status} - {result.message}"
            )

    return results
