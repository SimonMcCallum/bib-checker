"""Verify bib entries exist via CrossRef and Semantic Scholar."""

import re
import time
import requests
from rich.console import Console
from rich.table import Table

console = Console()

CROSSREF_URL = "https://api.crossref.org/works"
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"
OPENALEX_URL = "https://api.openalex.org/works"

HEADERS = {
    "User-Agent": "bib-checker/0.1 (https://github.com/SimonMcCallum/bib-checker)"
}


def _decode_inverted_abstract(inv: dict | None) -> str:
    """OpenAlex returns abstracts as {word: [positions]}; turn it back into text."""
    if not inv:
        return ""
    positions = []
    for word, idxs in inv.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def search_openalex(title: str, doi: str | None = None) -> dict | None:
    """Search OpenAlex. Often has abstracts where Semantic Scholar doesn't,
    especially for older / book-chapter / non-DOI works."""
    try:
        if doi:
            resp = requests.get(
                f"{OPENALEX_URL}/doi:{doi}", headers=HEADERS, timeout=15
            )
            if resp.status_code == 200:
                item = resp.json()
                return {
                    "source": "openalex",
                    "doi": (item.get("doi") or "").replace("https://doi.org/", ""),
                    "title": item.get("title", ""),
                    "year": str(item.get("publication_year", "")),
                    "abstract": _decode_inverted_abstract(item.get("abstract_inverted_index")),
                }

        params = {"search": title, "per-page": 3}
        resp = requests.get(OPENALEX_URL, params=params, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return None
        items = resp.json().get("results", [])
        for item in items:
            oa_title = item.get("title", "") or ""
            if _title_match(title, oa_title) > 0.6:
                return {
                    "source": "openalex",
                    "doi": (item.get("doi") or "").replace("https://doi.org/", ""),
                    "title": oa_title,
                    "year": str(item.get("publication_year", "")),
                    "abstract": _decode_inverted_abstract(item.get("abstract_inverted_index")),
                }
    except (requests.RequestException, KeyError, IndexError):
        pass
    return None


def _normalize(s: str) -> str:
    """Lowercase, strip punctuation and extra spaces."""
    s = re.sub(r"[^a-z0-9\s]", "", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _title_match(a: str, b: str) -> float:
    """Jaccard similarity of word sets."""
    wa = set(_normalize(a).split())
    wb = set(_normalize(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def search_crossref(title: str, doi: str = None) -> dict | None:
    """Search CrossRef for a paper. Returns best match or None."""
    try:
        if doi:
            resp = requests.get(
                f"{CROSSREF_URL}/{doi}", headers=HEADERS, timeout=15
            )
            if resp.status_code == 200:
                item = resp.json()["message"]
                return {
                    "source": "crossref",
                    "doi": item.get("DOI", ""),
                    "title": " ".join(item.get("title", [""])),
                    "year": str(
                        item.get("published-print", {})
                        .get("date-parts", [[None]])[0][0]
                        or item.get("created", {})
                        .get("date-parts", [[None]])[0][0]
                        or ""
                    ),
                }

        params = {"query.title": title, "rows": 3}
        resp = requests.get(
            CROSSREF_URL, params=params, headers=HEADERS, timeout=15
        )
        if resp.status_code != 200:
            return None

        items = resp.json().get("message", {}).get("items", [])
        for item in items:
            cr_title = " ".join(item.get("title", [""]))
            if _title_match(title, cr_title) > 0.6:
                return {
                    "source": "crossref",
                    "doi": item.get("DOI", ""),
                    "title": cr_title,
                    "year": str(
                        item.get("published-print", {})
                        .get("date-parts", [[None]])[0][0]
                        or ""
                    ),
                }
    except (requests.RequestException, KeyError, IndexError):
        pass
    return None


def search_semantic_scholar(title: str) -> dict | None:
    """Search Semantic Scholar for a paper. Returns best match or None.

    Also returns the model-generated 'tldr' (1–2 sentence summary) when
    available — this is often crisper signal than the full abstract for
    claim-alignment scoring.
    """
    try:
        params = {
            "query": title,
            "limit": 3,
            "fields": "title,year,authors,abstract,tldr,externalIds",
        }
        resp = requests.get(
            S2_SEARCH_URL, params=params, headers=HEADERS, timeout=15
        )
        if resp.status_code != 200:
            return None

        papers = resp.json().get("data", [])
        for paper in papers:
            s2_title = paper.get("title", "")
            if _title_match(title, s2_title) > 0.6:
                tldr_obj = paper.get("tldr") or {}
                return {
                    "source": "semantic_scholar",
                    "paper_id": paper.get("paperId", ""),
                    "title": s2_title,
                    "year": str(paper.get("year", "")),
                    "authors": [
                        a.get("name", "") for a in paper.get("authors", [])
                    ],
                    "abstract": paper.get("abstract", ""),
                    "tldr": tldr_obj.get("text", "") if isinstance(tldr_obj, dict) else "",
                    "doi": (paper.get("externalIds") or {}).get("DOI", ""),
                }
    except (requests.RequestException, KeyError, IndexError):
        pass
    return None


def verify_entry(key: str, entry: dict) -> dict:
    """Verify a single bib entry. Returns result dict."""
    title = entry.get("title", "")
    doi = entry.get("doi", "")
    year = entry.get("year", "")

    result = {
        "key": key,
        "title": title,
        "bib_year": year,
        "found": False,
        "source": "",
        "matched_title": "",
        "matched_year": "",
        "year_match": False,
        "doi": "",
        "abstract": "",
        "tldr": "",
    }

    if not title:
        result["note"] = "No title in bib entry"
        return result

    # Try Semantic Scholar first (free, often has abstracts for indexed papers)
    s2 = search_semantic_scholar(title)
    if s2 and (s2.get("abstract") or s2.get("tldr")):
        result["found"] = True
        result["source"] = "semantic_scholar"
        result["matched_title"] = s2["title"]
        result["matched_year"] = s2["year"]
        result["year_match"] = s2["year"] == year
        result["doi"] = s2.get("doi", "")
        result["abstract"] = s2.get("abstract", "")
        result["tldr"] = s2.get("tldr", "")
        result["paper_id"] = s2.get("paper_id", "")
        return result

    time.sleep(1)

    # OpenAlex — best abstract coverage for older / non-DOI works
    oa = search_openalex(title, doi=doi)
    if oa and oa.get("abstract"):
        result["found"] = True
        result["source"] = "openalex"
        result["matched_title"] = oa["title"]
        result["matched_year"] = oa["year"]
        result["year_match"] = oa["year"] == year
        result["doi"] = oa.get("doi", "")
        result["abstract"] = oa.get("abstract", "")
        return result

    # Either S2 or OpenAlex found the paper but neither had an abstract.
    # Take whichever metadata we have rather than failing over to CrossRef alone.
    if s2 or oa:
        meta = s2 or oa
        result["found"] = True
        result["source"] = meta["source"]
        result["matched_title"] = meta["title"]
        result["matched_year"] = meta["year"]
        result["year_match"] = meta["year"] == year
        result["doi"] = meta.get("doi", "")
        result["abstract"] = ""
        return result

    time.sleep(1)

    # Final fallback: CrossRef metadata only (no abstracts)
    cr = search_crossref(title, doi=doi)
    if cr:
        result["found"] = True
        result["source"] = "crossref"
        result["matched_title"] = cr["title"]
        result["matched_year"] = cr["year"]
        result["year_match"] = cr["year"] == year
        result["doi"] = cr.get("doi", "")
        return result

    result["note"] = "Not found in Semantic Scholar, OpenAlex, or CrossRef"
    return result


def verify_all(entries: dict, delay: float = 1.0) -> list[dict]:
    """Verify all entries with rate limiting. Returns list of results."""
    results = []
    total = len(entries)

    for i, (key, entry) in enumerate(entries.items(), 1):
        title = entry.get("title", "(no title)")
        console.print(f"[dim][{i}/{total}][/dim] {key}: {title[:60]}...")
        result = verify_entry(key, entry)

        status = "[green]FOUND[/green]" if result["found"] else "[red]NOT FOUND[/red]"
        year_info = ""
        if result["found"] and not result["year_match"]:
            year_info = f" [yellow](year: bib={result['bib_year']}, found={result['matched_year']})[/yellow]"
        console.print(f"  {status}{year_info}")

        results.append(result)
        if i < total:
            time.sleep(delay)

    return results


def print_summary(results: list[dict]):
    """Print a summary table of verification results."""
    table = Table(title="Verification Summary")
    table.add_column("Key", style="cyan")
    table.add_column("Status")
    table.add_column("Source")
    table.add_column("Year Match")
    table.add_column("DOI")

    found = 0
    for r in results:
        if r["found"]:
            found += 1
            status = "[green]Found[/green]"
            year = "[green]Yes[/green]" if r["year_match"] else "[yellow]No[/yellow]"
        else:
            status = "[red]Not found[/red]"
            year = ""
        table.add_row(
            r["key"],
            status,
            r.get("source", ""),
            year,
            r.get("doi", "")[:40],
        )

    console.print(table)
    console.print(f"\n[bold]Found: {found}/{len(results)}[/bold]")
