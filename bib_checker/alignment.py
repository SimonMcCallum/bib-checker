"""Check whether citation context aligns with the cited paper's abstract."""

import re
from rich.console import Console
from rich.table import Table

console = Console()


def _clean_text(text: str) -> str:
    """Remove LaTeX commands and normalize whitespace, lowercased.

    Lowercasing is fine for embedding similarity and lexical scorers, where
    case doesn't carry meaning. Do NOT use this for NLI input — cross-encoder
    NLI models are case-sensitive; use _clean_text_keepcase instead.
    """
    text = re.sub(r"\\[a-zA-Z]+\*?\s*", " ", text)
    text = re.sub(r"[{}$~\\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _clean_text_keepcase(text: str) -> str:
    """Same as _clean_text but preserves case — for NLI inputs."""
    text = re.sub(r"\\[a-zA-Z]+\*?\s*", " ", text)
    text = re.sub(r"[{}$~\\]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def _isolate_cite_sentence(context: str, cite_key: str) -> str:
    """Return just the sentence containing the citation marker.

    The parser stores ~200 chars on each side of a `\\cite{}`, which often
    spans multiple sentences. For NLI we want only the sentence carrying the
    actual claim — including neighbours muddies the entailment judgment.
    """
    if not context:
        return ""
    # Find the citation marker position. Try the bib key, then bare 'cite'.
    if cite_key and cite_key in context:
        idx = context.index(cite_key)
    elif "cite" in context.lower():
        idx = context.lower().index("cite")
    else:
        return context
    sentences = _SENT_SPLIT.split(context)
    cursor = 0
    for s in sentences:
        end = cursor + len(s)
        if cursor <= idx <= end + 2:
            return s.strip()
        cursor = end + 1  # +1 for the split whitespace
    return context


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

    For each citation we already know (from the embedding step) the single
    sentence in the cited paper that best matches the surrounding claim. We
    feed *that* sentence to NLI as the premise — not every sentence in the
    abstract — because:
      1. NLI models perform best on a single coherent claim, not a paragraph.
      2. Aggregating max(contradiction) over many sentences will almost always
         find one neutral/off-topic sentence the model can flag, even when the
         paper supports the claim.
    If the embedding step didn't run (e.g. --scorer tfidf), we compute the
    best-matching snippet here on the fly.

    Mutates results in place, adding `nli_polarity`, `nli_label`, `nli_premise`.
    A citation is escalated to flagged=True when polarity exceeds `threshold`.
    """
    from .nli import score_pairs, DEFAULT_NLI_MODEL

    pairs: list[tuple[str, str]] = []
    targets: list[dict] = []  # results entries that will receive a verdict

    for r in results:
        entry = entries.get(r["key"], {})
        # Premise: the full claim of the cited paper. tldr is a model-generated
        # one-sentence summary, abstract is the author's claim. Joining them
        # gives NLI more substance than a single best-matching sentence and
        # lets it judge the paper's actual conclusion, not a tangential clause.
        tldr = (entry.get("tldr") or "").strip()
        abstract = (entry.get("abstract") or "").strip()
        premise_parts = [p for p in (tldr, abstract) if p]
        # Title-only premise produces unreliable NLI verdicts (a title isn't a
        # claim the model can entail/contradict). Skip those rather than
        # surface a misleading polarity score.
        if not premise_parts:
            r["nli_polarity"] = None
            r["nli_label"] = None
            r["nli_skipped_reason"] = "no abstract or tldr available"
            continue

        # Hypothesis: the single sentence around the citation marker, not the
        # 400-char window. The wider window includes neighbouring sentences
        # that don't actually carry the claim being cited.
        ctxs = r.get("contexts") or []
        if not ctxs:
            r["nli_polarity"] = None
            r["nli_label"] = None
            continue
        cite_key = r.get("key", "")
        hypothesis = _isolate_cite_sentence(ctxs[0], cite_key) or ctxs[0]

        # Preserve case — cross-encoder NLI models are case-sensitive.
        premise = _clean_text_keepcase(" ".join(premise_parts))[:1500]
        hypothesis = _clean_text_keepcase(hypothesis)[:1500]
        pairs.append((premise, hypothesis))
        targets.append(r)
        r["nli_premise"] = premise[:300]
        r["nli_hypothesis"] = hypothesis

    if not pairs:
        return

    raw = score_pairs(pairs, model_name=model_name or DEFAULT_NLI_MODEL)
    for r, scores in zip(targets, raw):
        polarity = scores["polarity"]
        # Pick the dominant label by raw probability
        label = max(("contradiction", "entailment", "neutral"), key=lambda L: scores[L])
        r["nli_polarity"] = polarity
        r["nli_label"] = label
        r["nli_scores"] = scores
        if polarity >= threshold:
            current = r.get("status", "ok")
            r["status"] = _bump_tier(current)
            r["flagged"] = r["status"] in ("check", "flag")
            extra = f"NLI polarity {polarity:.2f} (label={label})"
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


def _tier(score: float | None, ok_threshold: float, flag_threshold: float) -> str:
    """Three-tier status from a similarity score.

        score >= ok_threshold      -> "ok"
        flag_threshold <= score    -> "check"   (in between OK and FLAG)
        score < flag_threshold     -> "flag"
        score is None              -> "no_data"
    """
    if score is None:
        return "no_data"
    if score >= ok_threshold:
        return "ok"
    if score >= flag_threshold:
        return "check"
    return "flag"


def _bump_tier(tier: str) -> str:
    """Raise severity by one level, used when NLI signals contradiction."""
    return {"ok": "check", "check": "flag"}.get(tier, tier)


def check_alignment(
    citations: dict,
    entries: dict,
    threshold: float = 0.30,
    flag_threshold: float = 0.20,
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
                "status": "no_data",
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
        status = _tier(primary, threshold, flag_threshold)
        notes = []
        if title_only:
            notes.append("title only (no abstract/tldr)")
        if status == "flag":
            notes.append(f"low similarity ({scorer}={primary:.3f} < {flag_threshold})")
        elif status == "check":
            notes.append(f"borderline similarity ({scorer}={primary:.3f}, {flag_threshold}–{threshold})")

        results.append({
            "key": key,
            "title": title,
            "contexts": citations[key],
            "abstract": (joined_ref[:200] + "...") if len(joined_ref) > 200 else joined_ref,
            "keyword_score": keyword,
            "tfidf_score": tfidf,
            "embedding_score": emb,
            "best_snippet": best_snip,
            "status": status,
            "flagged": status in ("check", "flag"),  # backward-compat
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

    counts = {"ok": 0, "check": 0, "flag": 0, "no_data": 0}

    for r in results:
        status = r.get("status") or ("no_data" if r.get("keyword_score") is None else "ok")
        counts[status] = counts.get(status, 0) + 1

        if status == "no_data":
            row = [r["key"], "—", "—", "—"]
            if has_nli:
                row.append("—")
            if has_llm:
                row.append("—")
            row += ["[dim]No data[/dim]", r.get("reason", "")]
            table.add_row(*row)
            continue

        kw = f"{r['keyword_score']:.3f}"
        tf = f"{r['tfidf_score']:.3f}" if r.get("tfidf_score") is not None else "—"
        em = f"{r['embedding_score']:.3f}" if r.get("embedding_score") is not None else "—"

        status_cell = {
            "ok":    "[green]OK[/green]",
            "check": "[yellow]CHECK[/yellow]",
            "flag":  "[red]FLAG[/red]",
        }[status]

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
        row += [status_cell, r.get("reason", "")]
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\n[bold]OK: {counts['ok']}, "
        f"[yellow]Check: {counts['check']}[/yellow], "
        f"[red]Flag: {counts['flag']}[/red], "
        f"No data: {counts['no_data']}[/bold]"
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
        s = r.get("status")
        status = {
            "ok": "OK", "check": "CHECK", "flag": "FLAG", "no_data": "No data",
        }.get(s, "OK" if r.get("keyword_score") is not None else "No data")
        note = r.get("reason", "")
        cols = [r["key"], kw, tf, em]
        if has_nli:
            pol = r.get("nli_polarity")
            cols.append(f"{pol:+.2f}" if pol is not None else "—")
        if has_llm:
            cols.append((r.get("llm_verdict") or {}).get("verdict", "—"))
        cols += [status, note]
        lines.append("| " + " | ".join(str(c) for c in cols) + " |")

        if r.get("flagged"):
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
