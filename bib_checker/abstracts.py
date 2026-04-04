"""Fetch abstracts and add them to bib entries."""

import time
from datetime import date
from rich.console import Console

from .parser import parse_bib, write_bib_field
from .verify import verify_entry

console = Console()


def fetch_and_add_abstracts(bib_path: str, delay: float = 1.0) -> dict:
    """Fetch abstracts for all entries and write them into the bib file.

    Returns summary stats: {added: int, skipped: int, not_found: int}.
    """
    entries = parse_bib(bib_path)
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

        if result["found"] and result["abstract"]:
            abstract = result["abstract"]
            write_bib_field(bib_path, key, "abstract", abstract)
            annotation = f"Verified via {result['source']} on {today}"
            if result.get("doi"):
                annotation += f". DOI: {result['doi']}"
            write_bib_field(bib_path, key, "annotation", annotation)

            console.print(f"  [green]Added abstract ({len(abstract)} chars)[/green]")
            stats["added"] += 1

            # Re-parse to pick up changes for subsequent entries
            entries = parse_bib(bib_path)
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
