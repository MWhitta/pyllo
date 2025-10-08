"""Helpers for interacting with CBORG metadata."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from io import StringIO
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup

CBORG_MODELS_URL = "https://cborg.lbl.gov/models/"


@dataclass
class CBORGModel:
    endpoint: str
    creator: str
    name: str
    context: str
    vision: str
    cost: str
    security: str
    api_names: List[str] = field(default_factory=list)


def _extract_api_name_map(soup: BeautifulSoup) -> dict[str, List[str]]:
    mapping: dict[str, List[str]] = {}
    for strong in soup.find_all("strong"):
        if "API Model Name" not in strong.get_text(strip=True):
            continue
        header = strong.find_previous(["h2", "h3", "h4"])
        if not header:
            continue
        key = header.get_text(strip=True).lower()
        codes = [code.get_text(strip=True) for code in strong.parent.find_all("code")]
        if not codes:
            text = strong.parent.get_text(" ", strip=True)
            suffix = text.split("API Model Name", 1)[-1]
            codes = [item.strip(" ,") for item in suffix.split(",") if item.strip()]
        if not codes:
            continue
        existing = mapping.setdefault(key, [])
        for code in codes:
            if code not in existing:
                existing.append(code)
    return mapping


def fetch_cborg_models(url: str = CBORG_MODELS_URL) -> List[CBORGModel]:
    """Fetch the CBORG models table and return a list of CBORGModel entries."""

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    api_map = _extract_api_name_map(soup)

    models: List[CBORGModel] = []
    tables = soup.find_all("table")
    expected = [
        "Model Endpoint Location",
        "Model Creator",
        "Model Name",
        "Context Length*",
        "Vision",
        "Cost**",
        "Security Level",
    ]
    for table in tables:
        headers = [th.get_text(strip=True) for th in table.find_all("th")]
        if headers[: len(expected)] != expected:
            continue

        tbody = table.find("tbody")
        if not tbody:
            continue
        for row in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cells) < len(expected):
                continue
            model = CBORGModel(
                endpoint=cells[0],
                creator=cells[1],
                name=cells[2],
                context=cells[3],
                vision=cells[4],
                cost=cells[5],
                security=cells[6],
            )
            key_candidates = {
                model.name.lower(),
                f"{model.creator} {model.name}".lower(),
            }
            api_names: List[str] = []
            for key, codes in api_map.items():
                if any(candidate in key for candidate in key_candidates) or any(
                    key in candidate for candidate in key_candidates
                ):
                    api_names.extend(codes)
            model.api_names = sorted(set(api_names))
            models.append(model)
    return models


def cborg_models_as_csv(models: Iterable[CBORGModel]) -> str:
    """Render CBORG model entries to CSV string."""

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ["Endpoint", "Creator", "Model", "API Names", "Context", "Vision", "Cost", "Security"]
    )
    for model in models:
        writer.writerow(
            [
                model.endpoint,
                model.creator,
                model.name,
                "; ".join(model.api_names) if model.api_names else "",
                model.context,
                model.vision,
                model.cost,
                model.security,
            ]
        )
    return output.getvalue()
