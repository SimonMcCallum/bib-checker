"""Check whether citation context aligns with the cited paper's abstract."""

import re
from rich.console import Console
from rich.table import Table

console = Console()


def _clean_text(text: str) -> str:
    """Remove LaTeX commands and normalize whitespace."""
    text = re.sub(r"\\[a-zA-Z]+\*?\s*", " ", text)
    text = re.sub(r"[{}$~\\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _keyword_overlap(text_a: str, text_b: str) -> float:
    """Simple keyword overlap score (Jaccard on word sets)."""
    # Remove stopwords
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "can",
        "this", "that", "these", "those", "it", "its", "we", "our",
        "they", "their", "not", "no", "as", "if", "than", "more",
    }
    words_a = set(_clean_text(text_a).split()) - stopwords
    words_b = set(_clean_text(text_b).split()) - stopwords
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    """TF-IDF cosine similarity between two texts."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        tfidf = vectorizer.fit_transform([_clean_text(text_a), _clean_text(text_b)])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return float(sim)
    except (ImportError, ValueError):
        return _keyword_overlap(text_a, text_b)


def check_alignment(
    citations: dict,
    entries: dict,
    threshold: float = 0.08,
) -> list[dict]:
    """Check alignment between citation contexts and abstracts.

    Args:
        citations: {key: [context_str, ...]} from parser.extract_citations
        entries: {key: {field: value}} from parser.parse_bib
        threshold: flag citations below this similarity score

    Returns list of alignment results.
    """
    results = []

    for key, contexts in citations.items():
        entry = entries.get(key, {})
        abstract = entry.get("abstract", "")
        title = entry.get("title", "(no title)")

        if not abstract:
            results.append({
                "key": key,
                "title": title,
                "contexts": contexts,
                "abstract": "",
                "keyword_score": None,
                "tfidf_score": None,
                "flagged": False,
                "reason": "no abstract available",
            })
            continue

        # Combine all contexts for this citation
        combined_context = " ".join(contexts)

        keyword = _keyword_overlap(combined_context, abstract)
        tfidf = _tfidf_similarity(combined_context, abstract)

        flagged = tfidf < threshold
        reason = ""
        if flagged:
            reason = f"Low similarity (tfidf={tfidf:.3f} < {threshold})"

        results.append({
            "key": key,
            "title": title,
            "contexts": contexts,
            "abstract": abstract[:200] + "..." if len(abstract) > 200 else abstract,
            "keyword_score": keyword,
            "tfidf_score": tfidf,
            "flagged": flagged,
            "reason": reason,
        })

    return results


def print_alignment(results: list[dict]):
    """Print alignment results as a rich table."""
    table = Table(title="Citation Alignment Check")
    table.add_column("Key", style="cyan", max_width=25)
    table.add_column("Keyword", justify="right")
    table.add_column("TF-IDF", justify="right")
    table.add_column("Status")
    table.add_column("Note", max_width=40)

    flagged_count = 0
    checked = 0

    for r in results:
        if r["keyword_score"] is None:
            table.add_row(
                r["key"], "—", "—", "[dim]No abstract[/dim]", r.get("reason", "")
            )
            continue

        checked += 1
        kw = f"{r['keyword_score']:.3f}"
        tf = f"{r['tfidf_score']:.3f}"

        if r["flagged"]:
            flagged_count += 1
            status = "[red]FLAG[/red]"
        else:
            status = "[green]OK[/green]"

        table.add_row(r["key"], kw, tf, status, r.get("reason", ""))

    console.print(table)
    console.print(
        f"\n[bold]Checked: {checked}, Flagged: {flagged_count}, "
        f"No abstract: {len(results) - checked}[/bold]"
    )


def generate_report(
    results: list[dict],
    output_path: str,
    tex_path: str = "",
    bib_path: str = "",
):
    """Generate a markdown report of alignment results."""
    lines = [
        "# Citation Alignment Report",
        "",
        f"- **Paper:** {tex_path}",
        f"- **Bibliography:** {bib_path}",
        f"- **Entries checked:** {len(results)}",
        "",
        "## Summary",
        "",
        "| Key | Keyword | TF-IDF | Status | Note |",
        "|-----|---------|--------|--------|------|",
    ]

    flagged_entries = []
    for r in results:
        kw = f"{r['keyword_score']:.3f}" if r["keyword_score"] is not None else "—"
        tf = f"{r['tfidf_score']:.3f}" if r["tfidf_score"] is not None else "—"
        status = "FLAG" if r["flagged"] else ("OK" if r["keyword_score"] is not None else "No abstract")
        note = r.get("reason", "")
        lines.append(f"| {r['key']} | {kw} | {tf} | {status} | {note} |")

        if r["flagged"]:
            flagged_entries.append(r)

    if flagged_entries:
        lines.extend(["", "## Flagged Citations", ""])
        for r in flagged_entries:
            lines.append(f"### {r['key']}")
            lines.append(f"**Title:** {r['title']}")
            lines.append(f"**TF-IDF:** {r['tfidf_score']:.3f}")
            lines.append("")
            lines.append("**Citation context(s):**")
            for ctx in r["contexts"]:
                lines.append(f"> {ctx[:300]}")
            lines.append("")
            if r["abstract"]:
                lines.append(f"**Abstract:** {r['abstract']}")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"\n[bold]Report written to {output_path}[/bold]")
