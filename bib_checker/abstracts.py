"""Fetch abstracts and add them to bib entries."""

import time
from datetime import date
from rich.console import Console

from .parser import parse_bib, write_bib_field
from .verify import verify_entry

console = Console()


def fetch_and_add_abstracts(
    bib_path: str,
    delay: float = 1.0,
    cited_keys: set | None = None,
) -> dict:
    """Fetch abstracts (and Semantic Scholar tldrs) and write them into the bib.

    If `cited_keys` is given, only entries whose keys are in that set will be
    fetched — useful when a personal library is much larger than the citations
    actually used by the paper under review.

    Returns summary stats: {added: int, skipped: int, not_found: int}.
    """
    all_entries = parse_bib(bib_path)
    if cited_keys is not None:
        entries = {k: v for k, v in all_entries.items() if k in cited_keys}
    else:
        entries = all_entries
    stats = {"added": 0, "skipped": 0, "not_found": 0, "total": len(entries)}
    today = date.today().isoformat()

    for i, (key, entry) in enumerate(entries.items(), 1):
        title = entry.get("title", "(no title)")
        console.print(f"[dim][{i}/{len(entries)}][/dim] {key}: {title[:60]}...")

        # Skip if abstract already exists
        if entry.get("abstract"):
            console.print("  [dim]Already has abstract, skipping[/dim]")
            stats["skipped"] += 1
            continue

        result = verify_entry(key, entry)

        if result["found"] and (result.get("abstract") or result.get("tldr")):
            abstract = result.get("abstract", "")
            tldr = result.get("tldr", "")
            if abstract:
                write_bib_field(bib_path, key, "abstract", abstract)
            if tldr:
                write_bib_field(bib_path, key, "tldr", tldr)
            annotation = f"Verified via {result['source']} on {today}"
            if result.get("doi"):
                annotation += f". DOI: {result['doi']}"
            write_bib_field(bib_path, key, "annotation", annotation)

            tldr_note = f" + tldr ({len(tldr)} chars)" if tldr else ""
            abs_note = f"abstract ({len(abstract)} chars)" if abstract else "tldr only"
            console.print(f"  [green]Added {abs_note}{tldr_note}[/green]")
            stats["added"] += 1

            # Re-parse to pick up changes for subsequent entries
            all_entries = parse_bib(bib_path)
            entries = (
                {k: v for k, v in all_entries.items() if k in cited_keys}
                if cited_keys is not None else all_entries
            )
        elif result["found"]:
            console.print("  [yellow]Found but no abstract available[/yellow]")
            annotation = f"Verified via {result['source']} on {today} (no abstract)"
            write_bib_field(bib_path, key, "annotation", annotation)
            stats["skipped"] += 1
        else:
            console.print("  [red]Not found in any database[/red]")
            stats["not_found"] += 1

        if i < len(entries):
            time.sleep(delay)

    console.print(
        f"\n[bold]Abstracts: {stats['added']} added, "
        f"{stats['skipped']} skipped, {stats['not_found']} not found "
        f"(out of {stats['total']})[/bold]"
    )
    return stats
