"""CLI entry point for bib-checker."""

import argparse
import sys
from rich.console import Console

from .parser import parse_bib, extract_citations
from .verify import verify_all, print_summary
from .abstracts import fetch_and_add_abstracts
from .alignment import check_alignment, print_alignment, generate_report

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


def cmd_check(args):
    """Check citation-abstract alignment."""
    entries = parse_bib(args.bib)
    citations = extract_citations(args.tex)

    console.print(
        f"[bold]Checking {len(citations)} cited keys "
        f"against {len(entries)} bib entries[/bold]\n"
    )

    results = check_alignment(citations, entries, threshold=args.threshold)
    print_alignment(results)


def cmd_report(args):
    """Generate a full alignment report."""
    entries = parse_bib(args.bib)
    citations = extract_citations(args.tex)

    console.print(
        f"[bold]Generating report for {len(citations)} citations[/bold]\n"
    )

    # First verify and fetch abstracts if needed
    no_abstract = [k for k, e in entries.items() if not e.get("abstract") and k in citations]
    if no_abstract and not args.skip_fetch:
        console.print(
            f"[yellow]{len(no_abstract)} cited entries lack abstracts. "
            f"Fetching...[/yellow]\n"
        )
        fetch_and_add_abstracts(args.bib, delay=args.delay)
        entries = parse_bib(args.bib)  # Re-parse after adding abstracts

    results = check_alignment(citations, entries, threshold=args.threshold)
    print_alignment(results)

    output = args.output or "alignment_report.md"
    generate_report(results, output, tex_path=args.tex, bib_path=args.bib)


def main():
    parser = argparse.ArgumentParser(
        prog="bib-checker",
        description="Verify bibliography entries and check citation alignment.",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay between API requests in seconds (default: 1.0)",
    )
    sub = parser.add_subparsers(dest="command")

    # verify
    p_verify = sub.add_parser("verify", help="Verify bib entries exist")
    p_verify.add_argument("bib", help="Path to .bib file")

    # abstracts
    p_abs = sub.add_parser("abstracts", help="Fetch and add abstracts")
    p_abs.add_argument("bib", help="Path to .bib file")

    # check
    p_check = sub.add_parser("check", help="Check citation alignment")
    p_check.add_argument("tex", help="Path to .tex file")
    p_check.add_argument("bib", help="Path to .bib file")
    p_check.add_argument(
        "--threshold", type=float, default=0.08,
        help="TF-IDF similarity threshold for flagging (default: 0.08)",
    )

    # report
    p_report = sub.add_parser("report", help="Generate alignment report")
    p_report.add_argument("tex", help="Path to .tex file")
    p_report.add_argument("bib", help="Path to .bib file")
    p_report.add_argument("-o", "--output", help="Output markdown file")
    p_report.add_argument(
        "--threshold", type=float, default=0.08,
        help="TF-IDF similarity threshold for flagging (default: 0.08)",
    )
    p_report.add_argument(
        "--skip-fetch", action="store_true",
        help="Skip fetching missing abstracts",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "verify": cmd_verify,
        "abstracts": cmd_abstracts,
        "check": cmd_check,
        "report": cmd_report,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
