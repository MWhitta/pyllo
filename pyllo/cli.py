"""Command-line interface for Pyllo."""

from __future__ import annotations

from pathlib import Path
from typing import List

import typer
from rich.console import Console
from rich.table import Table

from .cborg import fetch_cborg_models
from .config import Settings
from .ingest import ingest_corpus
from .minerals import collect_mineral_manuscripts
from .rag import ClayRAG
from .structures import StructureDownloaderError, gather_structures

app = typer.Typer(help="Pyllo CLI: clay-science retrieval augmented generation toolkit.")
console = Console()


@app.command()
def ingest(
    data_dir: Path = typer.Option(None, help="Override data directory for vector store output."),
    corpus_dir: Path = typer.Option(None, help="Override literature directory containing PDFs."),
) -> None:
    """Ingest PDFs into the local vector store."""
    settings = Settings()
    if data_dir:
        settings.data_dir = data_dir
    if corpus_dir:
        settings.corpus_dirs = [corpus_dir]

    console.print("[bold cyan]Starting ingestion[/bold cyan]")
    store_path = ingest_corpus(settings)
    console.print(f"[green]Ingestion complete. Vector store at {store_path}[/green]")
    console.print(
        "[yellow]Press Ctrl+C if the shell prompt does not reappear after ingestion.[/yellow]"
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Clay-science question to send to the RAG assistant."),
    top_k: int = typer.Option(None, help="Override number of retrieved chunks."),
    show_context: bool = typer.Option(True, help="Display supporting context after the answer."),
) -> None:
    """Ask the Pyllo RAG assistant a question."""
    settings = Settings()
    try:
        rag = ClayRAG(settings)
    except FileNotFoundError as exc:
        console.print(
            "[red]Vector store not found. Run `pyllo ingest` first to build "
            "the knowledge base.[/red]"
        )
        raise typer.Exit(code=1) from exc

    if top_k:
        rag.retriever.retriever_config.top_k = top_k

    response = rag.answer(question)
    console.print(f"[bold green]Answer:[/bold green] {response.answer}\n")

    if show_context:
        table = Table(title="Retrieved Context", show_header=True, header_style="bold magenta")
        table.add_column("Citation", style="cyan", justify="left")
        table.add_column("Preview", style="white", justify="left")
        for ctx in response.context:
            if "\n" in ctx:
                citation, content = ctx.split("\n", 1)
            else:
                citation, content = "Context", ctx
            table.add_row(citation, content[:240] + ("..." if len(content) > 240 else ""))
        if table.row_count:
            console.print(table)
        else:
            console.print("[yellow]No supporting context retrieved.[/yellow]")


@app.command("minerals-download")
def minerals_download(
    minerals: List[str] = typer.Option(
        None, "--mineral", "-m", help="Specific mineral names to process (repeatable)."
    ),
    mineral_dir: Path = typer.Option(
        None, help="Path to mineral CSV exports (default: data/minerals)."
    ),
    output_dir: Path = typer.Option(None, help="Directory to store downloaded manuscripts."),
    max_per_mineral: int = typer.Option(3, help="Maximum manuscripts to collect for each mineral."),
    crossref_rows: int = typer.Option(
        12, help="Number of Crossref results to pull before filtering."
    ),
    sleep_seconds: float = typer.Option(1.0, help="Delay between Crossref requests."),
    dry_run: bool = typer.Option(False, help="Only gather metadata without downloading PDFs."),
) -> None:
    """Collect manuscripts for IMA minerals and download available PDFs."""

    results = collect_mineral_manuscripts(
        minerals=minerals,
        mineral_dir=mineral_dir,
        output_dir=output_dir,
        max_per_mineral=max_per_mineral,
        crossref_rows=crossref_rows,
        sleep_seconds=sleep_seconds,
        download=not dry_run,
    )

    unique_minerals = {item.mineral for item in results}
    console.print(
        f"[green]Collected {len(results)} manuscripts across {len(unique_minerals)} minerals."
    )


@app.command("cborg-models")
def cborg_models(
    show_details: bool = typer.Option(False, help="Include additional columns in the listing.")
) -> None:
    """Print the list of CBORG models available via the OpenAI-compatible endpoint."""

    models = fetch_cborg_models()
    if not models:
        console.print(
            "[yellow]No CBORG models found. Check https://cborg.lbl.gov/models/ "
            "for updates.[/yellow]"
        )
        raise typer.Exit(1)

    table = Table(title="CBORG Models", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("API Name(s)", style="green")
    table.add_column("Creator")
    table.add_column("Endpoint")
    if show_details:
        table.add_column("Context")
        table.add_column("Vision")
        table.add_column("Cost")
        table.add_column("Security")

    for model in models:
        api_display = ", ".join(model.api_names) if model.api_names else "—"
        row = [model.name, api_display, model.creator, model.endpoint]
        if show_details:
            row.extend([model.context, model.vision, model.cost, model.security])
        table.add_row(*row)

    console.print(table)


@app.command("structures-download")
def structures_download(
    csv_path: Path = typer.Option(
        None,
        "--csv",
        help=(
            "Path to the RRUFF mineral CSV export (defaults to latest "
            "rag-minerals-rruff-export-*.csv)."
        ),
    ),
    minerals: List[str] = typer.Option(
        None,
        "--mineral",
        "-m",
        help="Limit downloads to one or more mineral names (repeatable).",
    ),
    limit: int = typer.Option(0, help="Stop after processing N minerals (0 processes all)."),
    skip_experimental: bool = typer.Option(
        False, help="Skip experimental CIF downloads from RRUFF."
    ),
    skip_simulated: bool = typer.Option(
        False, help="Skip simulated CIF downloads from Materials Project."
    ),
    api_key: str = typer.Option(
        None,
        "--materials-api-key",
        "-k",
        help="Materials Project API key (falls back to MAPI_KEY env var).",
    ),
    sleep_seconds: float = typer.Option(0.5, help="Delay between HTTP requests."),
) -> None:
    """Download experimental (RRUFF) and simulated (Materials Project) CIFs into data/structure."""

    if not csv_path:
        minerals_dir = Path("data/minerals")
        matches = sorted(minerals_dir.glob("rag-minerals-rruff-export-*.csv"))
        if not matches:
            console.print(
                "[red]Mineral CSV not found. Provide --csv or place a "
                "rag-minerals-rruff-export-*.csv under data/minerals.[/red]"
            )
            raise typer.Exit(code=1)
        csv_path = matches[-1]

    try:
        results = gather_structures(
            csv_path=csv_path,
            base_dir=Path("data"),
            minerals=minerals,
            limit=limit or None,
            include_experimental=not skip_experimental,
            include_simulated=not skip_simulated,
            api_key=api_key,
            sleep_seconds=sleep_seconds,
            console=console,
        )
    except StructureDownloaderError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    downloaded = [res for res in results if res.status == "downloaded"]
    existing = [res for res in results if res.status == "exists"]
    skipped = [res for res in results if res.status not in {"downloaded", "exists"}]

    console.print(
        f"[green]Downloaded {len(downloaded)} structures[/green], "
        f"[yellow]{len(existing)} already present[/yellow], "
        f"[cyan]{len(skipped)} skipped or missing[/cyan]."
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
