"""Suggest citations for sentences in a LaTeX document.

The inverse of `bib_checker.alignment`: instead of asking "does this citation
support its surrounding claim", this module asks "what citation(s) from the
bib would best support each sentence?". A sentence with a strong match but
no existing citation is the most actionable signal — the writer either
forgot to cite a known reference or has unintentionally re-derived something
already in their library.

Approach:
    1. Walk the .tex file/folder, split into sentences, and record any
       \\cite{} keys already on each sentence.
    2. Encode every sentence with the same sentence-transformer used for
       alignment. Encode every bib entry the same way (tldr + abstract or
       title fallback).
    3. For each sentence, compute cosine similarity against every bib entry
       and keep the top-k.
    4. Flag sentences whose top-1 score crosses a confidence threshold but
       have no citation, *or* whose top-1 isn't already cited there.

The default threshold is calibrated empirically: 0.55 on `all-MiniLM-L6-v2`
puts you near the upper tail of "the cited paper genuinely supports this
claim" matches. Lower it for more suggestions / higher noise; raise it to
only see strong matches.
"""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.table import Table

from . import embed

console = Console()

# Sentence splitter and LaTeX cleanup tuned for academic prose
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\\])")
_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?(\s*\[[^\]]*\])?\s*(\{[^}]*\})?")
_INLINE_MATH = re.compile(r"\$[^$]*\$")
_DISPLAY_MATH = re.compile(r"\\\[.*?\\\]|\\\(.*?\\\)", re.DOTALL)
_ENV_BLOCK = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|figure\*?|table\*?|tabular|verbatim|lstlisting|tikzpicture)\}"
    r".*?"
    r"\\end\{\1\}",
    re.DOTALL,
)
_CITE_CMD = re.compile(r"\\cite[pt]?\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}")


def _strip_latex(text: str) -> str:
    text = _ENV_BLOCK.sub(" ", text)
    text = _DISPLAY_MATH.sub(" ", text)
    text = _INLINE_MATH.sub(" ", text)
    # Drop comments
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
    return text


def _clean_sentence(sentence: str) -> str:
    """Strip LaTeX commands but keep words, for embedding input."""
    s = _LATEX_CMD.sub(" ", sentence)
    s = re.sub(r"[{}~\\]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_sentences(tex_path: str,
                      exclude_dirs: tuple = ("Figures", "figures"),
                      min_words: int = 8) -> list[dict]:
    """Extract sentences from a .tex file or recursively from a directory.

    Each result is a dict with:
        - file: str, the .tex file the sentence came from
        - idx: int, position within that file
        - text: str, the cleaned sentence text (for embedding)
        - raw: str, the original sentence including LaTeX (for display)
        - cite_keys: list[str], any \\cite keys already inside the sentence
    Sentences shorter than `min_words` after cleanup are dropped — usually
    headings, list markers, or label/ref glue.
    """
    p = Path(tex_path)
    if p.is_file():
        files = [p]
    else:
        files = [
            t for t in p.rglob("*.tex")
            if not any(part in exclude_dirs for part in t.relative_to(p).parts)
        ]

    out: list[dict] = []
    for tex_file in files:
        try:
            text = tex_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        stripped = _strip_latex(text)
        # Split on sentence boundaries
        for idx, raw in enumerate(_SENT_SPLIT.split(stripped)):
            raw = raw.strip()
            if not raw:
                continue
            cite_keys: list[str] = []
            for m in _CITE_CMD.finditer(raw):
                cite_keys.extend(k.strip() for k in m.group(1).split(","))
            cleaned = _clean_sentence(raw)
            if len(cleaned.split()) < min_words:
                continue
            out.append({
                "file": str(tex_file),
                "idx": idx,
                "text": cleaned,
                "raw": raw,
                "cite_keys": cite_keys,
            })
    return out


def _entry_text(entry: dict) -> str:
    parts = []
    if entry.get("tldr"):
        parts.append(entry["tldr"].strip())
    if entry.get("abstract"):
        parts.append(entry["abstract"].strip())
    if not parts and entry.get("title"):
        parts.append(entry["title"].strip())
    return " ".join(parts)


def suggest_citations(
    sentences: list[dict],
    entries: dict,
    threshold: float = 0.55,
    top_k: int = 3,
    only_missing: bool = True,
) -> list[dict]:
    """For each sentence, return top-k bib entries by cosine similarity.

    `only_missing=True` (default) keeps only sentences whose best match is not
    already cited in that sentence — i.e., the actionable cases. Set False
    to also see "you cited X but Y is a better fit" sentences.
    """
    # Build the bib library
    keys: list[str] = []
    texts: list[str] = []
    for k, e in entries.items():
        t = _entry_text(e)
        if t:
            keys.append(k)
            texts.append(t)
    if not keys:
        return []

    console.print(f"[dim]Encoding {len(keys)} bib entries...[/dim]")
    bib_vecs = embed.encode(texts)

    if not sentences:
        return []
    console.print(f"[dim]Encoding {len(sentences)} sentences...[/dim]")
    sent_vecs = embed.encode([s["text"] for s in sentences])

    # Cosine similarity matrix (vectors are L2-normalised, dot = cosine)
    scores = sent_vecs @ bib_vecs.T  # (S, B)

    out = []
    for i, sent in enumerate(sentences):
        row = scores[i]
        # Top-k by score
        top_idx = np.argsort(-row)[:top_k]
        suggestions = [
            {"key": keys[int(j)], "score": float(row[int(j)])}
            for j in top_idx
        ]
        best = suggestions[0] if suggestions else None
        if best is None or best["score"] < threshold:
            continue
        # Only flag sentences where the best match isn't already cited there
        if only_missing and best["key"] in (sent.get("cite_keys") or []):
            continue
        out.append({
            **sent,
            "suggestions": suggestions,
            "best_score": best["score"],
            "missing": best["key"] not in (sent.get("cite_keys") or []),
        })
    return out


def write_markdown_report(suggestions: list[dict],
                           entries: dict,
                           out_path: str,
                           threshold: float) -> None:
    """Write a markdown report grouped by source file."""
    by_file: dict[str, list[dict]] = {}
    for s in suggestions:
        by_file.setdefault(s["file"], []).append(s)

    lines = [
        "# Citation Suggestions",
        "",
        f"Sentences below have at least one bib entry whose abstract/tldr "
        f"scores **>= {threshold:.2f}** semantic similarity, but the top match "
        f"isn't currently cited in the sentence.",
        "",
        f"**Total flagged sentences:** {len(suggestions)}",
        "",
    ]
    for file_path in sorted(by_file):
        lines.append(f"## {Path(file_path).name}")
        lines.append("")
        for s in by_file[file_path]:
            lines.append(f"### Sentence {s['idx']} (score {s['best_score']:.3f})")
            lines.append("")
            lines.append(f"> {s['raw'][:600]}")
            lines.append("")
            existing = s.get("cite_keys") or []
            if existing:
                lines.append(f"**Already cites:** {', '.join(existing)}")
            else:
                lines.append("**No existing citation in this sentence.**")
            lines.append("")
            lines.append("**Top suggestions:**")
            for sug in s["suggestions"]:
                title = entries.get(sug["key"], {}).get("title", "")
                marker = " (already cited here)" if sug["key"] in existing else ""
                lines.append(f"- `{sug['key']}` — {sug['score']:.3f} — {title}{marker}")
            lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[bold]Wrote {len(suggestions)} suggestions to {out_path}[/bold]")


def print_summary(suggestions: list[dict], top_n: int = 25) -> None:
    """Print the strongest suggestions to the console as a table."""
    table = Table(title=f"Top {top_n} citation suggestions (by score)")
    table.add_column("Score", justify="right", style="cyan")
    table.add_column("File")
    table.add_column("Suggested key", style="magenta")
    table.add_column("Sentence preview", max_width=60)
    sorted_sugs = sorted(suggestions, key=lambda x: -x["best_score"])[:top_n]
    for s in sorted_sugs:
        table.add_row(
            f"{s['best_score']:.3f}",
            Path(s["file"]).name,
            s["suggestions"][0]["key"],
            s["raw"][:200].replace("\n", " "),
        )
    console.print(table)
    console.print(f"\n[bold]Suggestions with score >= threshold: {len(suggestions)}[/bold]")
