"""Suggest citations from a personal bib library given a query passage.

This is a small RAG-for-citations: encode each entry in the user's bib
(abstract preferred, title fallback) once, then for any query string return
the top-k most-similar entries. Everything runs locally.
"""

from pathlib import Path
import json
import numpy as np
from rich.console import Console
from rich.table import Table

from . import embed
from .parser import parse_bib

console = Console()


def _entry_text(entry: dict) -> str:
    abstract = entry.get("abstract", "").strip()
    title = entry.get("title", "").strip()
    if abstract and title:
        return f"{title}. {abstract}"
    return abstract or title


def build_library(bib_path: str, cache_path: str | None = None) -> dict:
    """Build (or load) an embedding library for a bib file.

    Returns dict with keys: vectors (N, dim), keys (list[str]), entries (dict).
    If cache_path is given and exists with the same bib mtime, the cache is reused.
    """
    bib = Path(bib_path)
    if not bib.exists():
        raise FileNotFoundError(bib_path)

    cache = Path(cache_path) if cache_path else None
    if cache and cache.exists():
        meta = json.loads(cache.with_suffix(".meta.json").read_text())
        if meta.get("bib_mtime") == bib.stat().st_mtime and meta.get("bib_path") == str(bib.resolve()):
            data = np.load(cache)
            return {
                "vectors": data["vectors"],
                "keys": list(meta["keys"]),
                "entries": parse_bib(bib_path),
            }

    entries = parse_bib(bib_path)
    keys, texts = [], []
    for key, entry in entries.items():
        text = _entry_text(entry)
        if text:
            keys.append(key)
            texts.append(text)

    if not keys:
        raise ValueError(f"No usable entries (need title or abstract) in {bib_path}")

    console.print(f"[dim]Encoding {len(keys)} entries...[/dim]")
    vectors = embed.encode(texts)

    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, vectors=vectors)
        cache.with_suffix(".meta.json").write_text(json.dumps({
            "bib_mtime": bib.stat().st_mtime,
            "bib_path": str(bib.resolve()),
            "keys": keys,
            "model": embed.DEFAULT_MODEL,
        }))

    return {"vectors": vectors, "keys": keys, "entries": entries}


def suggest(library: dict, query: str, k: int = 5) -> list[dict]:
    """Return top-k bib entries most similar to the query string."""
    qv = embed.encode([query])[0]
    scores = library["vectors"] @ qv
    top = np.argsort(-scores)[:k]
    out = []
    for idx in top:
        key = library["keys"][int(idx)]
        entry = library["entries"][key]
        out.append({
            "key": key,
            "score": float(scores[int(idx)]),
            "title": entry.get("title", "(no title)"),
            "author": entry.get("author", ""),
            "year": entry.get("year", ""),
            "abstract": entry.get("abstract", ""),
        })
    return out


def print_suggestions(query: str, suggestions: list[dict]):
    console.print(f"\n[bold]Query:[/bold] {query[:200]}")
    table = Table(title="Citation suggestions")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("Key", style="magenta")
    table.add_column("Year")
    table.add_column("Title", max_width=70)
    for s in suggestions:
        table.add_row(
            f"{s['score']:.3f}",
            s["key"],
            s["year"] or "—",
            s["title"][:120],
        )
    console.print(table)
