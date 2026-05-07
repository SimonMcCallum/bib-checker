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


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter — good enough for paper abstracts."""
    text = (text or "").strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    # If splitting failed (one big blob), keep the whole thing
    return parts or [text]


def _embedding_similarity_batch(
    items: list[tuple[str, list[str]]],
) -> list[tuple[float, str]]:
    """Sentence-level embedding similarity.

    items is a list of (ctx, ref_snippets) where ref_snippets is the cited
    paper broken into searchable units — typically [tldr, abstract_sent_1,
    abstract_sent_2, ...] so the score can match the *single* sentence in the
    abstract that supports the claim, instead of being averaged out by the
    rest.

    Returns a list of (best_score, best_snippet) per input item.
    """
    from . import embed
    if not items:
        return []

    contexts = [_clean_text(c) for c, _ in items]
    ctx_vecs = embed.encode(contexts)

    # Pack all snippets into one batch for a single forward pass
    all_snippets: list[str] = []
    offsets: list[tuple[int, int]] = []
    for _, snippets in items:
        cleaned = [_clean_text(s) for s in snippets if s and s.strip()]
        if not cleaned:
            offsets.append((-1, -1))
            continue
        offsets.append((len(all_snippets), len(all_snippets) + len(cleaned)))
        all_snippets.extend(cleaned)

    if all_snippets:
        snippet_vecs = embed.encode(all_snippets)
    else:
        snippet_vecs = None

    out = []
    for i, (start, end) in enumerate(offsets):
        if start < 0 or snippet_vecs is None:
            out.append((0.0, ""))
            continue
        scores = snippet_vecs[start:end] @ ctx_vecs[i]
        idx = int(scores.argmax())
        out.append((float(scores[idx]), items[i][1][idx]))
    return out


def _build_ref_snippets(entry: dict) -> tuple[list[str], bool]:
    """Pick the best representations of the cited paper for matching.

    Returns (snippets, used_title_only).
    Order: tldr first, then abstract-sentences. Title is the last-resort fallback.
    """
    tldr = (entry.get("tldr") or "").strip()
    abstract = (entry.get("abstract") or "").strip()
    title = (entry.get("title") or "").strip()

    snippets: list[str] = []
    if tldr:
        snippets.append(tldr)
    if abstract:
        snippets.extend(_split_sentences(abstract))
    if snippets:
        return snippets, False
    if title:
        return [title], True
    return [], True


def add_polarity_check(
    results: list[dict],
    citations: dict,
    entries: dict,
    threshold: float = 0.3,
    model_name: str | None = None,
) -> None:
    """Run a local NLI cross-encoder over each citation to detect polarity flips.

    Mutates results in place, adding `nli_polarity`, `nli_label`, `nli_premise`.
    A citation is escalated to flagged=True when polarity exceeds `threshold`.
    """
    from .nli import score_citation_polarity, DEFAULT_NLI_MODEL

    polarity = score_citation_polarity(
        {r["key"]: r["contexts"] for r in results if r.get("contexts")},
        entries,
        model_name=model_name or DEFAULT_NLI_MODEL,
    )
    for r in results:
        info = polarity.get(r["key"])
        if not info:
            r["nli_polarity"] = None
            r["nli_label"] = None
            continue
        r["nli_polarity"] = info["best_polarity"]
        r["nli_label"] = info["best_label"]
        r["nli_premise"] = info["best_premise"]
        if info["best_polarity"] >= threshold:
            r["flagged"] = True
            extra = f"NLI polarity {info['best_polarity']:.2f} (label={info['best_label']})"
            r["reason"] = (r.get("reason") or "") + ("; " if r.get("reason") else "") + extra


def llm_review_results(
    results: list[dict],
    entries: dict,
    llm_client,
    review_all: bool = False,
) -> None:
    """Mutate `results` in place, attaching an LLM verdict to each item.

    By default only items with flagged=True (or no abstract/title) are sent.
    Pass review_all=True to send every citation.
    """
    if llm_client is None:
        return
    for r in results:
        if not review_all and not r["flagged"]:
            r["llm_verdict"] = None
            continue
        entry = entries.get(r["key"], {})
        abstract = entry.get("abstract", "") or r.get("abstract", "")
        title = entry.get("title", r.get("title", ""))
        contexts = r.get("contexts") or []
        if not contexts:
            r["llm_verdict"] = None
            continue
        try:
            verdict = llm_client.review(
                context=" ".join(contexts)[:2000],
                title=title,
                abstract=abstract,
            )
        except Exception as e:  # network / rate-limit / parse errors shouldn't kill the run
            verdict = {"verdict": "unknown", "confidence": 0.0, "reason": f"error: {e}"}
        r["llm_verdict"] = verdict
        # Mismatch / tangential are stronger flags than embedding alone
        if verdict.get("verdict") in ("mismatch", "tangential") and not r["flagged"]:
            r["flagged"] = True
            r["reason"] = (r.get("reason") or "") + f"; LLM: {verdict.get('verdict')}"


def check_alignment(
    citations: dict,
    entries: dict,
    threshold: float = 0.08,
    scorer: str = "embedding",
) -> list[dict]:
    """Check alignment between citation contexts and abstracts.

    Args:
        citations: {key: [context_str, ...]} from parser.extract_citations
        entries: {key: {field: value}} from parser.parse_bib
        threshold: flag citations below this similarity score

    Returns list of alignment results.
    """
    results = []

    scoreable = []
    for key, contexts in citations.items():
        entry = entries.get(key, {})
        title = entry.get("title", "(no title)")
        snippets, title_only = _build_ref_snippets(entry)

        if not snippets:
            results.append({
                "key": key,
                "title": title,
                "contexts": contexts,
                "abstract": "",
                "keyword_score": None,
                "tfidf_score": None,
                "embedding_score": None,
                "best_snippet": "",
                "flagged": False,
                "reason": "no abstract, tldr, or title available",
            })
            continue

        combined_ctx = " ".join(contexts)
        scoreable.append((key, combined_ctx, snippets, title_only, title))

    embedding_results: list[tuple[float, str]] = []
    if scorer == "embedding" and scoreable:
        embedding_results = _embedding_similarity_batch(
            [(c, snips) for _, c, snips, _, _ in scoreable]
        )

    for i, (key, ctx, snippets, title_only, title) in enumerate(scoreable):
        joined_ref = " ".join(snippets)
        keyword = _keyword_overlap(ctx, joined_ref)
        tfidf = _tfidf_similarity(ctx, joined_ref) if scorer != "embedding" else None
        if scorer == "embedding":
            emb, best_snip = embedding_results[i]
        else:
            emb, best_snip = None, ""

        primary = emb if scorer == "embedding" else tfidf
        flagged = primary is not None and primary < threshold
        notes = []
        if title_only:
            notes.append("title only (no abstract/tldr)")
        if flagged:
            notes.append(f"low similarity ({scorer}={primary:.3f} < {threshold})")

        results.append({
            "key": key,
            "title": title,
            "contexts": citations[key],
            "abstract": (joined_ref[:200] + "...") if len(joined_ref) > 200 else joined_ref,
            "keyword_score": keyword,
            "tfidf_score": tfidf,
            "embedding_score": emb,
            "best_snippet": best_snip,
            "flagged": flagged,
            "reason": "; ".join(notes),
        })

    return results


def print_alignment(results: list[dict]):
    """Print alignment results as a rich table."""
    has_llm = any(r.get("llm_verdict") for r in results)
    has_nli = any(r.get("nli_polarity") is not None for r in results)

    table = Table(title="Citation Alignment Check")
    table.add_column("Key", style="cyan", max_width=25)
    table.add_column("Keyword", justify="right")
    table.add_column("TF-IDF", justify="right")
    table.add_column("Embed", justify="right")
    if has_nli:
        table.add_column("NLI Δ", justify="right", max_width=8)
    if has_llm:
        table.add_column("LLM", max_width=12)
    table.add_column("Status")
    table.add_column("Note", max_width=40)

    flagged_count = 0
    checked = 0

    for r in results:
        if r["keyword_score"] is None:
            row = [r["key"], "—", "—", "—"]
            if has_nli:
                row.append("—")
            if has_llm:
                row.append("—")
            row += ["[dim]No data[/dim]", r.get("reason", "")]
            table.add_row(*row)
            continue

        checked += 1
        kw = f"{r['keyword_score']:.3f}"
        tf = f"{r['tfidf_score']:.3f}" if r.get("tfidf_score") is not None else "—"
        em = f"{r['embedding_score']:.3f}" if r.get("embedding_score") is not None else "—"

        if r["flagged"]:
            flagged_count += 1
            status = "[red]FLAG[/red]"
        else:
            status = "[green]OK[/green]"

        row = [r["key"], kw, tf, em]
        if has_nli:
            pol = r.get("nli_polarity")
            row.append(f"{pol:+.2f}" if pol is not None else "—")
        if has_llm:
            v = r.get("llm_verdict") or {}
            verdict = v.get("verdict", "—")
            colour = {
                "support": "green", "tangential": "yellow",
                "mismatch": "red", "unknown": "dim",
            }.get(verdict, "")
            row.append(f"[{colour}]{verdict}[/{colour}]" if colour else verdict)
        row += [status, r.get("reason", "")]
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\n[bold]Checked: {checked}, Flagged: {flagged_count}, "
        f"No data: {len(results) - checked}[/bold]"
    )


def generate_report(
    results: list[dict],
    output_path: str,
    tex_path: str = "",
    bib_path: str = "",
    diagnostics: dict | None = None,
):
    """Generate a markdown report of alignment results."""
    has_llm = any(r.get("llm_verdict") for r in results)
    has_nli = any(r.get("nli_polarity") is not None for r in results)

    lines = [
        "# Citation Alignment Report",
        "",
        f"- **Paper:** {tex_path}",
        f"- **Bibliography:** {bib_path}",
        f"- **Entries checked:** {len(results)}",
        "",
    ]

    if diagnostics:
        lines += ["## Bibliography diagnostics", ""]
        junk = diagnostics.get("junk_entries") or []
        unused = diagnostics.get("unused_entries") or []
        unresolved = diagnostics.get("unresolved_inline") or []
        lines.append(f"- Junk entries (section headings parsed as references): **{len(junk)}**")
        lines.append(f"- Unused entries (in bibliography, never cited inline): **{len(unused)}**")
        lines.append(f"- Unresolved inline citations (cited but missing from references): **{len(unresolved)}**")
        if junk:
            lines += ["", "### Junk entries", ""]
            for j in junk:
                lines.append(f"- `{j['key']}` — {j['title']}")
        if unused:
            lines += ["", "### Unused entries", ""]
            for u in unused:
                lines.append(f"- `{u['key']}` — {u['title']}")
        if unresolved:
            lines += ["", "### Unresolved inline citations", ""]
            for u in unresolved[:40]:  # cap noise
                txt = u.get("text") or "(empty)"
                ctx = (u.get("context") or "").replace("\n", " ")
                lines.append(f"- **{txt}** — {ctx[:160]}")
            if len(unresolved) > 40:
                lines.append(f"- ... and {len(unresolved) - 40} more")
        lines.append("")

    lines += ["## Summary", ""]
    extra_cols = []
    if has_nli:
        extra_cols.append("NLI Δ")
    if has_llm:
        extra_cols.append("LLM")
    header_cols = ["Key", "Keyword", "TF-IDF", "Embed", *extra_cols, "Status", "Note"]
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")

    flagged_entries = []
    for r in results:
        kw = f"{r['keyword_score']:.3f}" if r["keyword_score"] is not None else "—"
        tf = f"{r['tfidf_score']:.3f}" if r.get("tfidf_score") is not None else "—"
        em = f"{r['embedding_score']:.3f}" if r.get("embedding_score") is not None else "—"
        status = "FLAG" if r["flagged"] else ("OK" if r["keyword_score"] is not None else "No data")
        note = r.get("reason", "")
        cols = [r["key"], kw, tf, em]
        if has_nli:
            pol = r.get("nli_polarity")
            cols.append(f"{pol:+.2f}" if pol is not None else "—")
        if has_llm:
            cols.append((r.get("llm_verdict") or {}).get("verdict", "—"))
        cols += [status, note]
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")

        if r["flagged"]:
            flagged_entries.append(r)

    if flagged_entries:
        lines.extend(["", "## Flagged Citations", ""])
        for r in flagged_entries:
            lines.append(f"### {r['key']}")
            lines.append(f"**Title:** {r['title']}")
            primary = r.get("embedding_score") if r.get("embedding_score") is not None else r.get("tfidf_score")
            if primary is not None:
                lines.append(f"**Score:** {primary:.3f}")
            if r.get("best_snippet"):
                lines.append(f"**Best-matching sentence in cited paper:** _{r['best_snippet'][:300]}_")
            if r.get("nli_polarity") is not None:
                lines.append(
                    f"**NLI:** label={r.get('nli_label')}, "
                    f"polarity={r['nli_polarity']:+.3f} "
                    f"(>0 means cited paper contradicts the claim)"
                )
                if r.get("nli_premise"):
                    lines.append(f"**Premise (cited paper):** _{r['nli_premise'][:250]}_")
            v = r.get("llm_verdict")
            if v:
                lines.append(
                    f"**LLM:** {v.get('verdict', '—')} "
                    f"(confidence {v.get('confidence', 0):.2f}) — {v.get('reason', '')}"
                )
            lines.append("")
            lines.append("**Citation context(s):**")
            for ctx in r["contexts"]:
                lines.append(f"> {ctx[:300]}")
            lines.append("")
            if r["abstract"]:
                lines.append(f"**Abstract/title:** {r['abstract']}")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    console.print(f"\n[bold]Report written to {output_path}[/bold]")
