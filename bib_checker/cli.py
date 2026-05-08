"""CLI entry point for bib-checker."""

import argparse
import sys
from pathlib import Path
from rich.console import Console

from .parser import (
    parse_bib, parse_bib_dir, find_bib_paths_from_tex,
    extract_citations, extract_citations_from_dir,
)
from .verify import verify_all, print_summary
from .abstracts import fetch_and_add_abstracts
from .alignment import check_alignment, llm_review_results, add_polarity_check, print_alignment, generate_report
from .llm_review import make_client

console = Console()


def cmd_verify(args):
    """Verify all bib entries exist."""
    entries = parse_bib(args.bib)
    console.print(f"[bold]Verifying {len(entries)} entries from {args.bib}[/bold]\n")
    results = verify_all(entries, delay=args.delay)
    print_summary(results)


def cmd_abstracts(args):
    """Fetch and add abstracts to bib entries."""
    console.print(f"[bold]Fetching abstracts for {args.bib}[/bold]\n")
    fetch_and_add_abstracts(args.bib, delay=args.delay)


def _maybe_run_nli(args, results, citations, entries):
    """If --check-polarity is set, run a local NLI scorer for claim inversion."""
    if not getattr(args, "check_polarity", False):
        return
    console.print(f"[bold]Running NLI claim-inversion check (cross-encoder)...[/bold]")
    add_polarity_check(
        results, citations, entries,
        threshold=args.polarity_threshold,
        model_name=getattr(args, "nli_model", None),
    )


def _maybe_run_llm(args, results, entries):
    """Build LLM client from args and review flagged citations (if --llm != none)."""
    provider = getattr(args, "llm", "none")
    if provider == "none":
        return
    client = make_client(provider, model=getattr(args, "llm_model", None),
                         base_url=getattr(args, "llm_base_url", None))
    review_all = getattr(args, "llm_all", False)
    targets = sum(1 for r in results if r.get("flagged")) if not review_all else len(results)
    console.print(f"[bold]Running LLM review ({provider}) on {targets} citation(s)...[/bold]")
    llm_review_results(results, entries, client, review_all=review_all)


def _load_citations(tex_arg: str) -> dict:
    """Resolve a .tex file or a directory of .tex files to a citation dict."""
    p = Path(tex_arg)
    if p.is_dir():
        console.print(f"[dim]Scanning {p} for .tex files...[/dim]")
        return extract_citations_from_dir(str(p))
    return extract_citations(str(p))


def _load_bib(bib_arg: str, tex_arg: str | None = None) -> tuple[dict, str]:
    """Resolve a .bib file or a directory of .bib files.

    Behaviour:
      1. If `bib_arg` points to a single .bib file, use just that.
      2. If `bib_arg` is a directory and `tex_arg` is given, prefer .bib files
         actually referenced via \\bibliography{} / \\addbibresource{} in the
         LaTeX source (these may live in subfolders).
      3. Falls back to merging every .bib under `bib_arg` if no references
         can be discovered from the .tex.

    Returns (entries, primary_path). The primary_path is used as the write
    target when --fetch-missing or --bib-out are enabled.
    """
    p = Path(bib_arg)
    if p.is_file():
        return parse_bib(str(p)), str(p)

    if not p.is_dir():
        raise FileNotFoundError(f"{bib_arg} not found")

    # Directory: list .bib files referenced by the LaTeX source (these may
    # live in subfolders) AND any other .bib files directly under the dir.
    # The .tex-referenced files take precedence (first-seen wins on key
    # collision); the extras just fill in gaps — useful when an augmented bib
    # (e.g. one written by --fetch-missing) hasn't been wired into the .tex
    # yet but still carries richer data.
    seen_paths: set = set()
    ordered: list[Path] = []
    tex_refs: list[Path] = []
    if tex_arg:
        tex_refs = find_bib_paths_from_tex(tex_arg)
        for b in tex_refs:
            key = b.resolve()
            if key not in seen_paths:
                seen_paths.add(key)
                ordered.append(b)
    extras = [b for b in sorted(p.rglob("*.bib")) if b.resolve() not in seen_paths]
    for b in extras:
        seen_paths.add(b.resolve())
        ordered.append(b)
    if not ordered:
        raise FileNotFoundError(f"No .bib files found under {p}")

    if tex_refs:
        console.print(
            f"[dim]Bib files referenced by .tex ({len(tex_refs)}):[/dim]"
        )
        for b in tex_refs:
            try:
                rel = b.relative_to(p)
            except ValueError:
                rel = b
            console.print(f"  [dim]- {rel}[/dim]")
    if extras:
        console.print(
            f"[dim]Other .bib files in the project ({len(extras)}, used as fallback content):[/dim]"
        )
        for b in extras:
            try:
                rel = b.relative_to(p)
            except ValueError:
                rel = b
            console.print(f"  [dim]- {rel}[/dim]")

    # Field-level merge: for each key, take the union of fields across all
    # .bib files. The first occurrence "wins" for any field it provides, but
    # later bibs can fill in fields the earlier ones lack — so a sparse
    # references.bib (titles only) gets enriched by an augmented hlcc_paper.bib
    # that has abstracts/tldrs.
    merged: dict = {}
    for bib_file in ordered:
        for k, v in parse_bib(str(bib_file)).items():
            if k not in merged:
                merged[k] = dict(v)
                continue
            existing = merged[k]
            for fk, fv in v.items():
                if fv and not existing.get(fk):
                    existing[fk] = fv
    primary = tex_refs[0] if tex_refs else ordered[0]
    return merged, str(primary)


def cmd_check(args):
    """Check citation-abstract alignment from a .tex source (file or directory)."""
    entries, _ = _load_bib(args.bib, tex_arg=args.tex)
    citations = _load_citations(args.tex)

    console.print(
        f"[bold]Checking {len(citations)} cited keys "
        f"against {len(entries)} bib entries[/bold]\n"
    )

    results = check_alignment(
        citations, entries,
        threshold=args.threshold,
        flag_threshold=args.flag_threshold,
        scorer=args.scorer,
    )
    _maybe_run_nli(args, results, citations, entries)
    _maybe_run_llm(args, results, entries)
    print_alignment(results)


def cmd_report(args):
    """Generate a full alignment report from a .tex source (file or directory)."""
    import shutil

    citations = _load_citations(args.tex)

    # Load the bib(s). When args.bib is a directory, every .bib in it is merged
    # (first-seen-wins). primary_bib is the file we use as the write target for
    # the --fetch-missing / --bib-out copy.
    entries, primary_bib = _load_bib(args.bib, tex_arg=args.tex)
    bib_path = primary_bib

    # If --bib-out is given, work on a copy so the original .bib is untouched.
    if args.bib_out and Path(args.bib_out) != Path(primary_bib):
        Path(args.bib_out).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(primary_bib, args.bib_out)
        bib_path = args.bib_out
        console.print(f"[dim]Copied {primary_bib} -> {bib_path}[/dim]")

    console.print(
        f"[bold]Generating report for {len(citations)} citations against {len(entries)} bib entries[/bold]\n"
    )

    no_abstract = [k for k, e in entries.items() if not e.get("abstract") and k in citations]
    if no_abstract and args.fetch_missing:
        cited = set(citations.keys()) if not getattr(args, "fetch_all", False) else None
        scope = f"{len(no_abstract)} cited" if cited else f"all {len(entries)}"
        console.print(
            f"[yellow]Fetching abstracts for {scope} entries into {bib_path}...[/yellow]\n"
        )
        fetch_and_add_abstracts(bib_path, delay=args.delay, cited_keys=cited)
        # Re-merge the directory (or just the file) so freshly-fetched abstracts
        # are visible alongside entries from sibling bibs.
        merged_now, _ = _load_bib(args.bib, tex_arg=args.tex)
        # Override with anything just written into the primary/output bib.
        merged_now.update(parse_bib(bib_path))
        entries = merged_now
    elif no_abstract:
        console.print(
            f"[dim]{len(no_abstract)} cited entries lack abstracts; falling back to title. "
            f"Pass --fetch-missing to populate them.[/dim]\n"
        )

    results = check_alignment(
        citations, entries,
        threshold=args.threshold,
        flag_threshold=args.flag_threshold,
        scorer=args.scorer,
    )
    _maybe_run_nli(args, results, citations, entries)
    _maybe_run_llm(args, results, entries)
    print_alignment(results)

    output = args.output or "alignment_report.md"
    generate_report(results, output, tex_path=args.tex, bib_path=bib_path)


def cmd_pdf_extract(args):
    """Extract a .bib from a PDF via GROBID."""
    from . import pdf_extract
    entries, citations, diagnostics = pdf_extract.extract_from_pdf(
        args.pdf,
        base_url=args.grobid_url,
        tei_out=args.tei_out,
    )
    out = args.output or (Path(args.pdf).with_suffix(".bib").name)
    pdf_extract.write_bib(entries, out)
    console.print(f"[bold green]Wrote {len(entries)} entries to {out}[/bold green]")


def cmd_pdf_check(args):
    """Full pipeline: PDF -> bib -> abstracts -> alignment -> (optional LLM) -> report."""
    from . import pdf_extract

    pdf = Path(args.pdf)
    bib_out = Path(args.bib_out or pdf.with_suffix(".bib").name)
    tei_out = args.tei_out or str(pdf.with_suffix(".tei.xml"))

    console.print(f"[bold]1/4 Extracting references from {pdf.name}...[/bold]")
    entries, citations, diagnostics = pdf_extract.extract_from_pdf(
        str(pdf), base_url=args.grobid_url, tei_out=tei_out
    )
    pdf_extract.write_bib(entries, str(bib_out))
    console.print(f"[dim]Wrote {bib_out}[/dim]\n")

    if args.fetch_missing:
        console.print(f"[bold]2/4 Fetching abstracts...[/bold]")
        fetch_and_add_abstracts(str(bib_out), delay=args.delay)
        entries = parse_bib(str(bib_out))
    else:
        console.print(f"[bold]2/4 Skipping abstract fetch (pass --fetch-missing to enable).[/bold]\n")

    console.print(f"\n[bold]3/4 Scoring citation alignment ({args.scorer})...[/bold]")
    results = check_alignment(
        citations, entries,
        threshold=args.threshold,
        flag_threshold=args.flag_threshold,
        scorer=args.scorer,
    )
    _maybe_run_nli(args, results, citations, entries)
    _maybe_run_llm(args, results, entries)
    print_alignment(results)

    output = args.output or "alignment_report.md"
    console.print(f"\n[bold]4/4 Writing report...[/bold]")
    generate_report(
        results, output, tex_path=str(pdf), bib_path=str(bib_out),
        diagnostics=diagnostics,
    )


def cmd_suggest_doc(args):
    """Walk a document and suggest citations for sentences that lack a strong one."""
    from . import suggest_doc

    entries, _ = _load_bib(args.bib, tex_arg=args.tex)
    sentences = suggest_doc.extract_sentences(args.tex, min_words=args.min_words)
    console.print(
        f"[bold]Scanning {len(sentences)} sentences against {len(entries)} bib entries[/bold]\n"
    )
    suggestions = suggest_doc.suggest_citations(
        sentences, entries,
        threshold=args.threshold,
        top_k=args.top_k,
        only_missing=not args.include_existing,
    )
    suggest_doc.print_summary(suggestions, top_n=args.preview)
    if args.output:
        suggest_doc.write_markdown_report(
            suggestions, entries, args.output, threshold=args.threshold,
        )


def cmd_suggest(args):
    """Suggest citations from a personal bib for a query passage."""
    from . import suggest as sug

    cache = args.cache or str(Path(args.bib).with_suffix(".embcache.npz"))
    library = sug.build_library(args.bib, cache_path=cache)

    if args.query:
        query = args.query
    elif args.query_file:
        query = Path(args.query_file).read_text(encoding="utf-8")
    else:
        console.print("[red]Provide --query or --query-file[/red]")
        sys.exit(1)

    suggestions = sug.suggest(library, query, k=args.k)
    sug.print_suggestions(query, suggestions)


def main():
    parser = argparse.ArgumentParser(
        prog="bib-checker",
        description="Verify bibliography entries, check citation alignment, suggest citations.",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay between API requests in seconds (default: 1.0)",
    )
    sub = parser.add_subparsers(dest="command")

    p_verify = sub.add_parser("verify", help="Verify bib entries exist")
    p_verify.add_argument("bib", help="Path to .bib file")

    p_abs = sub.add_parser("abstracts", help="Fetch and add abstracts")
    p_abs.add_argument("bib", help="Path to .bib file")

    def _add_nli_args(p):
        p.add_argument("--check-polarity", action="store_true",
                       help="Run a local NLI cross-encoder to detect claim-inversion "
                            "(citations where the cited paper's findings contradict the claim). "
                            "Requires bib-checker[nli] extras.")
        p.add_argument("--polarity-threshold", type=float, default=0.30,
                       help="Flag citation when (contradiction_prob - entailment_prob) "
                            "exceeds this value (default: 0.30)")
        p.add_argument("--nli-model",
                       help="Override NLI model name. Default: cross-encoder/nli-deberta-v3-base. "
                            "For lighter footprint use cross-encoder/nli-MiniLM2-L6-H768.")

    def _add_llm_args(p):
        p.add_argument("--llm", choices=["none", "ollama", "anthropic", "openai"],
                       default="none",
                       help="Optional LLM second pass on flagged citations. "
                            "'none' (default) keeps the run fully algorithmic. "
                            "'ollama' uses a locally-hosted model. "
                            "'anthropic'/'openai' send (context, title, abstract) to the cloud "
                            "and require ANTHROPIC_API_KEY / OPENAI_API_KEY env vars.")
        p.add_argument("--llm-model",
                       help="Model name (default: claude-haiku-4-5-20251001 / gpt-4o-mini / llama3.1)")
        p.add_argument("--llm-base-url",
                       help="Override base URL (Ollama default http://localhost:11434, "
                            "or any OpenAI-compatible endpoint)")
        p.add_argument("--llm-all", action="store_true",
                       help="Send EVERY citation to the LLM, not just flagged ones (slower / costlier)")

    p_check = sub.add_parser("check", help="Check citation alignment from .tex")
    p_check.add_argument("tex", help="Path to .tex file OR directory containing .tex files")
    p_check.add_argument("bib", help="Path to .bib file OR directory containing .bib files (all merged)")
    p_check.add_argument("--threshold", type=float, default=0.30,
                         help="OK floor: scores >= this are OK (default 0.30)")
    p_check.add_argument("--flag-threshold", type=float, default=0.20,
                         help="FLAG ceiling: scores < this are FLAG; "
                              "scores between flag-threshold and threshold are CHECK "
                              "(default 0.20)")
    p_check.add_argument("--scorer", choices=["embedding", "tfidf"], default="embedding",
                         help="Similarity method (default: embedding)")
    _add_nli_args(p_check)
    _add_llm_args(p_check)

    p_report = sub.add_parser("report", help="Generate alignment report from .tex")
    p_report.add_argument("tex", help="Path to .tex file OR directory containing .tex files")
    p_report.add_argument("bib", help="Path to .bib file OR directory containing .bib files (all merged)")
    p_report.add_argument("-o", "--output", help="Output markdown file")
    p_report.add_argument("--bib-out",
                          help="Write the augmented bib (with fetched abstracts/tldrs) "
                               "to this path instead of modifying the input bib in place.")
    p_report.add_argument("--threshold", type=float, default=0.30,
                          help="OK floor: scores >= this are OK (default 0.30)")
    p_report.add_argument("--flag-threshold", type=float, default=0.20,
                          help="FLAG ceiling: scores < this are FLAG; "
                               "scores in between are CHECK (default 0.20)")
    p_report.add_argument("--scorer", choices=["embedding", "tfidf"], default="embedding")
    p_report.add_argument("--fetch-missing", action="store_true",
                          help="Fetch abstracts for cited entries that don't already have one")
    p_report.add_argument("--fetch-all", action="store_true",
                          help="With --fetch-missing, also fetch for entries that aren't "
                               "cited in the .tex (slower; useful for personal libraries).")
    _add_nli_args(p_report)
    _add_llm_args(p_report)

    p_pdfx = sub.add_parser("pdf-extract", help="Extract bib from PDF via GROBID")
    p_pdfx.add_argument("pdf", help="Path to PDF")
    p_pdfx.add_argument("-o", "--output", help="Output .bib path")
    p_pdfx.add_argument("--grobid-url", default="http://localhost:8070")
    p_pdfx.add_argument("--tei-out", help="Save raw TEI XML for inspection")

    p_pdfc = sub.add_parser("pdf-check", help="PDF -> bib -> abstracts -> alignment report")
    p_pdfc.add_argument("pdf", help="Path to PDF")
    p_pdfc.add_argument("--bib-out", help="Where to write the extracted .bib")
    p_pdfc.add_argument("--tei-out", help="Where to save raw TEI XML")
    p_pdfc.add_argument("-o", "--output", help="Output markdown report")
    p_pdfc.add_argument("--grobid-url", default="http://localhost:8070")
    p_pdfc.add_argument("--threshold", type=float, default=0.30,
                        help="OK floor: scores >= this are OK (default 0.30)")
    p_pdfc.add_argument("--flag-threshold", type=float, default=0.20,
                        help="FLAG ceiling: scores < this are FLAG; "
                             "scores in between are CHECK (default 0.20)")
    p_pdfc.add_argument("--scorer", choices=["embedding", "tfidf"], default="embedding")
    p_pdfc.add_argument("--fetch-missing", action="store_true",
                        help="Fetch abstracts for entries that don't already have one")
    _add_nli_args(p_pdfc)
    _add_llm_args(p_pdfc)

    p_sd = sub.add_parser(
        "suggest-doc",
        help="Scan a document for sentences that lack a strong citation and suggest one",
    )
    p_sd.add_argument("tex", help=".tex file or directory containing .tex files")
    p_sd.add_argument("bib", help=".bib file OR directory containing .bib files")
    p_sd.add_argument("-o", "--output", help="Markdown report path")
    p_sd.add_argument("--threshold", type=float, default=0.55,
                      help="Suggest only when top-1 score >= this (default 0.55). "
                           "Lower for more suggestions, higher for stricter.")
    p_sd.add_argument("--top-k", type=int, default=3,
                      help="How many candidate citations to show per sentence (default 3)")
    p_sd.add_argument("--min-words", type=int, default=8,
                      help="Drop sentences shorter than this many words (default 8)")
    p_sd.add_argument("--include-existing", action="store_true",
                      help="Also include sentences whose top suggestion is already cited there "
                           "(off by default — only the actionable cases are reported)")
    p_sd.add_argument("--preview", type=int, default=25,
                      help="How many top suggestions to print to the console (default 25)")

    p_sug = sub.add_parser("suggest", help="Suggest citations from a personal bib")
    p_sug.add_argument("bib", help="Personal .bib library")
    p_sug.add_argument("--query", help="Passage to find citations for")
    p_sug.add_argument("--query-file", help="File containing the query passage")
    p_sug.add_argument("-k", type=int, default=5, help="Number of suggestions")
    p_sug.add_argument("--cache", help="Embedding cache path (default: alongside bib)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "verify": cmd_verify,
        "abstracts": cmd_abstracts,
        "check": cmd_check,
        "report": cmd_report,
        "pdf-extract": cmd_pdf_extract,
        "pdf-check": cmd_pdf_check,
        "suggest": cmd_suggest,
        "suggest-doc": cmd_suggest_doc,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
