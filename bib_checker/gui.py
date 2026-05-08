"""Streamlit GUI for bib-checker.

Run with::

    streamlit run -m bib_checker.gui
    # or after install:
    bib-checker-gui

The UI lets the user pick:

  * Input mode  -- a single .tex file, a directory of .tex files, or a PDF.
  * The .bib file (suggested next to the input).
  * Where to write the updated .bib + report.
  * Checkboxes for: scorer choice, fetch-missing-abstracts, NLI polarity check,
    LLM provider, GROBID URL.

Everything runs in-process so progress is visible in the browser.
"""

from __future__ import annotations
import os
import sys
import traceback
from pathlib import Path

# `streamlit run gui.py` executes this file as a top-level script, so the
# `bib_checker` package must be importable by absolute name. Prepend the
# parent directory in case the user hasn't `pip install -e .`'d the project.
_PKG_PARENT = str(Path(__file__).resolve().parent.parent)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)


def _import_streamlit():
    try:
        import streamlit as st
        return st
    except ImportError:
        sys.stderr.write(
            "streamlit is not installed. Install the GUI extras with:\n"
            "    pip install bib-checker[gui]\n"
        )
        sys.exit(1)


def _pick_path(kind: str, title: str, filetypes: list | None = None) -> str:
    """Open a native OS picker. kind is 'file', 'folder', or 'save'.

    Streamlit runs locally for this tool, so tkinter dialogs appear on the
    same machine as the user's browser. Returns "" if the user cancels.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return ""

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        root.update()
    except Exception:
        pass

    if kind == "folder":
        path = filedialog.askdirectory(title=title, parent=root)
    elif kind == "save":
        path = filedialog.asksaveasfilename(
            title=title, parent=root, filetypes=filetypes or [],
            defaultextension=(filetypes[0][1].split()[0].lstrip("*") if filetypes else None),
        )
    else:
        path = filedialog.askopenfilename(
            title=title, parent=root, filetypes=filetypes or [],
        )
    root.destroy()
    return path or ""


def _path_input(st, label: str, key: str, kind: str,
                filetypes: list | None = None,
                default: str = "",
                help: str | None = None,
                button_label: str = "Browse") -> str:
    """Text input + Browse button that opens a native picker.

    Uses a pending-key trick so we can update the input value from a button
    without violating Streamlit's "don't mutate a live widget" rule.
    """
    pending_key = f"_{key}_pending"
    if pending_key in st.session_state:
        st.session_state[key] = st.session_state.pop(pending_key)
    elif key not in st.session_state and default:
        st.session_state[key] = default

    col1, col2 = st.columns([5, 1])
    with col1:
        value = st.text_input(label, key=key, help=help)
    with col2:
        # Vertical alignment with the text input
        st.write("")
        if st.button(button_label, key=f"{key}_browse_btn", width="stretch"):
            picked = _pick_path(kind, title=label)
            if picked:
                st.session_state[pending_key] = picked
                st.rerun()
    return value


def main():
    st = _import_streamlit()
    st.set_page_config(page_title="bib-checker", layout="wide")
    st.title("📚 bib-checker")
    st.caption(
        "Check whether your citations actually support the claims you make. "
        "Runs locally; nothing leaves your machine unless you opt in."
    )

    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------
    with st.sidebar:
        st.header("Input")
        mode = st.radio(
            "Source",
            ["LaTeX file", "LaTeX folder (recursive)", "PDF (via GROBID)"],
            help=(
                "LaTeX is the lightest option (works offline, no Docker). "
                "Folder mode walks subdirectories for chapter .tex files. "
                "PDF mode requires a running GROBID container."
            ),
        )

        if mode == "LaTeX file":
            tex_path = _path_input(
                st, "Path to .tex file", key="tex_path",
                kind="file", filetypes=[("LaTeX", "*.tex"), ("All files", "*.*")],
            )
        elif mode == "LaTeX folder (recursive)":
            tex_path = _path_input(
                st, "Path to LaTeX folder", key="tex_path",
                kind="folder",
                help="All .tex files under this folder will be scanned (Figures/ excluded).",
            )
        else:
            tex_path = _path_input(
                st, "Path to PDF", key="tex_path",
                kind="file", filetypes=[("PDF", "*.pdf"), ("All files", "*.*")],
            )

        # Suggest a .bib path next to the input
        suggested_bib = ""
        if tex_path:
            p = Path(tex_path)
            if p.is_dir():
                bibs = list(p.glob("*.bib"))
                if bibs:
                    suggested_bib = str(bibs[0])
            elif p.is_file():
                bibs = list(p.parent.glob("*.bib"))
                if bibs:
                    suggested_bib = str(bibs[0])
                elif mode == "PDF (via GROBID)":
                    suggested_bib = str(p.with_suffix(".bib"))

        # In PDF mode, the bib is an output (save dialog). Otherwise, an existing file.
        bib_kind = "save" if mode == "PDF (via GROBID)" else "file"
        bib_path = _path_input(
            st, "Path to .bib file", key="bib_path",
            kind=bib_kind,
            filetypes=[("BibTeX", "*.bib"), ("All files", "*.*")],
            default=suggested_bib,
            help="For PDF mode, this is where the extracted bib will be saved.",
        )

        report_default = ""
        if tex_path:
            p = Path(tex_path)
            if p.is_dir():
                report_default = str(p / "alignment_report.md")
            else:
                report_default = str(p.with_suffix(".alignment.md"))
        report_path = _path_input(
            st, "Output report (.md)", key="report_path",
            kind="save",
            filetypes=[("Markdown", "*.md"), ("All files", "*.*")],
            default=report_default,
        )

        st.markdown("---")
        st.header("Options")

        # Scoring
        scorer = st.selectbox(
            "Similarity scorer",
            ["embedding", "tfidf"],
            help="embedding: local sentence-transformer (semantic). tfidf: lexical, no model needed.",
        )
        threshold = st.slider("Flag below similarity", 0.0, 1.0, 0.30, 0.01)

        # Toggles
        fetch_missing = st.checkbox(
            "Fetch missing abstracts (Semantic Scholar / OpenAlex / CrossRef)",
            value=False,
            help="Sends only titles, not paper content. Already-populated abstracts are kept.",
        )
        check_polarity = st.checkbox(
            "Run NLI claim-inversion check (local cross-encoder)",
            value=False,
            help="Detects citations where the cited paper's findings contradict the claim. "
                 "Local model, ~440MB on first download, slow on first run.",
        )
        polarity_threshold = st.slider(
            "NLI polarity flag threshold", 0.0, 1.0, 0.30, 0.05,
            disabled=not check_polarity,
        )

        # LLM
        llm_provider = st.selectbox(
            "LLM second-pass review",
            ["none", "ollama", "anthropic", "openai"],
            help="'none' (default) keeps the run fully algorithmic. The cloud "
                 "providers send (citation context + title + abstract) and require "
                 "ANTHROPIC_API_KEY / OPENAI_API_KEY environment variables.",
        )
        llm_model = st.text_input(
            "LLM model (optional)",
            value="",
            disabled=(llm_provider == "none"),
            placeholder="claude-haiku-4-5-20251001 / gpt-4o-mini / llama3.1",
        )
        llm_all = st.checkbox(
            "Send every citation to the LLM (not just flagged ones)",
            value=False,
            disabled=(llm_provider == "none"),
        )

        # GROBID URL (only relevant in PDF mode)
        grobid_url = st.text_input(
            "GROBID URL",
            value="http://localhost:8070",
            disabled=(mode != "PDF (via GROBID)"),
        )

        run_button = st.button("Run check", type="primary", width="stretch")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    if not run_button:
        st.info("Configure the run in the sidebar and click **Run check**.")
        return

    # Widget keys (tex_path / bib_path / report_path) already persist across
    # reruns via Streamlit's session_state, so no explicit copy is needed here.

    if not tex_path or not Path(tex_path).exists():
        st.error("Source file/folder not found.")
        return

    # Absolute imports — `streamlit run gui.py` executes this file as a
    # top-level script, so relative imports (`from .parser ...`) would fail.
    from bib_checker.parser import (
        parse_bib, extract_citations, extract_citations_from_dir,
    )
    from bib_checker.verify import verify_all  # noqa: F401 (kept for parity)
    from bib_checker.abstracts import fetch_and_add_abstracts
    from bib_checker.alignment import (
        check_alignment, llm_review_results, add_polarity_check,
        generate_report,
    )
    from bib_checker.llm_review import make_client

    progress = st.progress(0, text="Starting...")
    log = st.expander("Log", expanded=True)
    diagnostics: dict | None = None

    try:
        # Stage 1: extract / parse citations
        if mode == "PDF (via GROBID)":
            from bib_checker import pdf_extract
            progress.progress(10, text="Extracting bibliography from PDF (GROBID)...")
            entries, citations, diagnostics = pdf_extract.extract_from_pdf(
                tex_path, base_url=grobid_url
            )
            if bib_path:
                pdf_extract.write_bib(entries, bib_path)
                log.write(f"Wrote {len(entries)} entries to {bib_path}")
        else:
            progress.progress(10, text="Parsing bibliography...")
            if not bib_path or not Path(bib_path).exists():
                st.error("Bibliography file not found.")
                return
            entries = parse_bib(bib_path)
            log.write(f"Parsed {len(entries)} bib entries")

            progress.progress(20, text="Extracting citation contexts...")
            if mode == "LaTeX folder (recursive)":
                citations = extract_citations_from_dir(tex_path)
            else:
                citations = extract_citations(tex_path)
            log.write(
                f"Found {sum(len(v) for v in citations.values())} citations "
                f"across {len(citations)} unique keys"
            )

        # Stage 2: optional abstract fetch
        if fetch_missing and bib_path:
            progress.progress(35, text="Fetching missing abstracts...")
            fetch_and_add_abstracts(bib_path, delay=1.0)
            entries = parse_bib(bib_path)

        # Stage 3: alignment
        progress.progress(60, text=f"Scoring alignment ({scorer})...")
        results = check_alignment(citations, entries, threshold=threshold, scorer=scorer)

        # Stage 4: optional NLI polarity check
        if check_polarity:
            progress.progress(75, text="Running NLI polarity check (local cross-encoder)...")
            add_polarity_check(results, citations, entries, threshold=polarity_threshold)

        # Stage 5: optional LLM second pass
        if llm_provider != "none":
            progress.progress(85, text=f"LLM review ({llm_provider})...")
            client = make_client(llm_provider, model=(llm_model or None))
            llm_review_results(results, entries, client, review_all=llm_all)

        # Stage 6: report
        progress.progress(95, text="Writing report...")
        if report_path:
            generate_report(
                results, report_path,
                tex_path=tex_path, bib_path=bib_path,
                diagnostics=diagnostics,
            )
            log.write(f"Report written to {report_path}")

        progress.progress(100, text="Done.")

        # ----- display -----
        flagged = [r for r in results if r.get("flagged")]
        st.subheader(f"Results: {len(results)} cited keys, {len(flagged)} flagged")

        # Build a plain dict-of-rows for the dataframe
        rows = []
        for r in results:
            rows.append({
                "key": r["key"],
                "embed": r.get("embedding_score"),
                "tfidf": r.get("tfidf_score"),
                "nli_polarity": r.get("nli_polarity"),
                "nli_label": r.get("nli_label"),
                "llm": (r.get("llm_verdict") or {}).get("verdict") if r.get("llm_verdict") else None,
                "flagged": r.get("flagged", False),
                "title": r.get("title", ""),
                "note": r.get("reason", ""),
            })
        st.dataframe(rows, width="stretch", height=400)

        if diagnostics:
            with st.expander(
                f"Bibliography diagnostics — "
                f"{len(diagnostics.get('junk_entries', []))} junk, "
                f"{len(diagnostics.get('unused_entries', []))} unused, "
                f"{len(diagnostics.get('unresolved_inline', []))} unresolved"
            ):
                if diagnostics.get("junk_entries"):
                    st.write("**Junk entries (section headings)**:",
                             diagnostics["junk_entries"])
                if diagnostics.get("unused_entries"):
                    st.write("**Unused entries (in bib, never cited)**:",
                             diagnostics["unused_entries"])
                if diagnostics.get("unresolved_inline"):
                    st.write("**Unresolved inline citations (cited, missing from bib)**:",
                             diagnostics["unresolved_inline"])

        if flagged:
            with st.expander(f"Flagged citations ({len(flagged)})", expanded=True):
                for r in flagged:
                    st.markdown(f"### `{r['key']}`")
                    st.markdown(f"**Title**: {r['title']}")
                    if r.get("embedding_score") is not None:
                        st.markdown(f"**Embedding similarity**: {r['embedding_score']:.3f}")
                    if r.get("nli_polarity") is not None:
                        st.markdown(
                            f"**NLI**: {r['nli_label']} "
                            f"(polarity {r['nli_polarity']:+.3f})"
                        )
                    v = r.get("llm_verdict")
                    if v:
                        st.markdown(
                            f"**LLM verdict**: {v.get('verdict')} — {v.get('reason')}"
                        )
                    if r.get("best_snippet"):
                        st.markdown(
                            f"**Best-matching sentence in cited paper**: _{r['best_snippet']}_"
                        )
                    st.markdown("**Citation context(s)**:")
                    for ctx in r["contexts"][:3]:
                        st.markdown(f"> {ctx[:400]}")
                    st.markdown("---")

        if report_path and Path(report_path).exists():
            st.download_button(
                "Download markdown report",
                Path(report_path).read_text(encoding="utf-8"),
                file_name=Path(report_path).name,
                mime="text/markdown",
            )

    except Exception as e:
        progress.empty()
        st.error(f"Run failed: {e}")
        st.code(traceback.format_exc())


def launch():
    """Entry point for `bib-checker-gui` — spawns `streamlit run` on this file."""
    import subprocess
    script_path = Path(__file__).resolve()
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(script_path)],
            check=True,
        )
    except FileNotFoundError:
        sys.stderr.write(
            "streamlit is not installed. Install the GUI extras with:\n"
            "    pip install bib-checker[gui]\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
