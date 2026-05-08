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

        # Suggest the project folder so EVERY .bib in it gets merged by
        # default. Users can browse to a single .bib to narrow it down,
        # or tick "Restrict to .tex-referenced bibs" below.
        suggested_bib = ""
        if tex_path:
            p = Path(tex_path)
            if p.is_dir():
                suggested_bib = str(p)
            elif p.is_file():
                suggested_bib = str(p.parent)
                if mode == "PDF (via GROBID)":
                    suggested_bib = str(p.with_suffix(".bib"))

        # In PDF mode, the bib is an output (save dialog). Otherwise, allow
        # either a single .bib file *or* a folder containing .bib files.
        # Default is the project folder so every .bib gets merged.
        if mode == "PDF (via GROBID)":
            bib_path = _path_input(
                st, "Path to .bib file", key="bib_path", kind="save",
                filetypes=[("BibTeX", "*.bib"), ("All files", "*.*")],
                default=suggested_bib,
                help="Where the extracted bib will be saved.",
            )
        else:
            bib_path = _path_input(
                st, "Path to .bib file or folder", key="bib_path", kind="folder",
                default=suggested_bib,
                help="Default is the project folder — every .bib file under it is merged "
                     "(so abstracts in one bib fill in titles from another). Browse opens "
                     "a folder picker; you can also paste a path to a single .bib file to "
                     "use just that one.",
                button_label="Browse folder",
            )

        tex_bib_only = False
        if mode != "PDF (via GROBID)":
            tex_bib_only = st.checkbox(
                "Restrict to .bib files referenced by the .tex",
                value=False,
                help="Off (default): merge every .bib under the folder. "
                     "On: only use .bib files explicitly referenced via "
                     r"\bibliography{} or \addbibresource{} in the LaTeX.",
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

        st.subheader("Citation alignment")
        scorer = st.selectbox(
            "Similarity scorer",
            ["embedding", "tfidf"],
            help="embedding: local sentence-transformer (semantic). tfidf: lexical, no model needed.",
        )
        ok_threshold = st.slider(
            "OK floor", 0.0, 1.0, 0.30, 0.01,
            help="Scores >= this are OK.",
        )
        flag_threshold = st.slider(
            "FLAG ceiling", 0.0, 1.0, 0.20, 0.01,
            help="Scores < this are FLAG. Scores between FLAG ceiling and OK floor are CHECK.",
        )

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

        st.subheader("Citation suggestions")
        suggest_missing = st.checkbox(
            "Scan document for sentences that lack a strong citation",
            value=False,
            help="Walks every sentence in the source, encodes it, and suggests "
                 "the best-matching bib entries. Highlights sentences whose top "
                 "match isn't currently cited there. LaTeX modes only.",
            disabled=(mode == "PDF (via GROBID)"),
        )
        suggestion_threshold = st.slider(
            "Suggestion threshold (top-1 score must exceed)",
            0.0, 1.0, 0.55, 0.01,
            disabled=not suggest_missing,
            help="Lower for more (noisier) suggestions; higher for stricter.",
        )
        suggestion_top_k = st.slider(
            "Candidates to show per sentence", 1, 5, 3, 1,
            disabled=not suggest_missing,
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
        parse_bib, load_bibs_smart, extract_citations, extract_citations_from_dir,
    )
    from bib_checker.verify import verify_all  # noqa: F401 (kept for parity)
    from bib_checker.abstracts import fetch_and_add_abstracts
    from bib_checker.alignment import (
        check_alignment, llm_review_results, add_polarity_check,
        generate_report,
    )
    from bib_checker.llm_review import make_client
    from bib_checker import suggest_doc as sd

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
                st.error("Bibliography file/folder not found.")
                return
            # Multi-bib aware load: when bib_path is a directory, merges all
            # .tex-referenced bibs plus any others under it (field-level merge).
            entries, primary_bib = load_bibs_smart(bib_path, tex_arg=tex_path, tex_only=tex_bib_only)
            log.write(f"Parsed {len(entries)} bib entries (primary: {primary_bib})")

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
            progress.progress(35, text="Fetching missing abstracts (cited entries only)...")
            cited = set(citations.keys()) if citations else None
            fetch_target = primary_bib if mode != "PDF (via GROBID)" else bib_path
            fetch_and_add_abstracts(fetch_target, delay=1.0, cited_keys=cited)
            entries, _ = load_bibs_smart(bib_path, tex_arg=tex_path, tex_only=tex_bib_only) if mode != "PDF (via GROBID)" else (parse_bib(bib_path), bib_path)

        # Stage 3: alignment
        progress.progress(60, text=f"Scoring alignment ({scorer})...")
        results = check_alignment(
            citations, entries,
            threshold=ok_threshold,
            flag_threshold=flag_threshold,
            scorer=scorer,
        )

        # Stage 4: optional NLI polarity check
        if check_polarity:
            progress.progress(75, text="Running NLI polarity check (local cross-encoder)...")
            add_polarity_check(results, citations, entries, threshold=polarity_threshold)

        # Stage 5: optional LLM second pass
        if llm_provider != "none":
            progress.progress(85, text=f"LLM review ({llm_provider})...")
            client = make_client(llm_provider, model=(llm_model or None))
            llm_review_results(results, entries, client, review_all=llm_all)

        # Stage 6: optional citation-suggestion pass
        suggestions: list[dict] = []
        if suggest_missing and mode != "PDF (via GROBID)":
            progress.progress(90, text="Scanning sentences for missing citations...")
            sentences = sd.extract_sentences(tex_path)
            log.write(
                f"Extracted {len(sentences)} sentences for suggestion scoring"
            )
            suggestions = sd.suggest_citations(
                sentences, entries,
                threshold=suggestion_threshold,
                top_k=suggestion_top_k,
                only_missing=True,
            )

        # Stage 7: report
        progress.progress(95, text="Writing report...")
        if report_path:
            generate_report(
                results, report_path,
                tex_path=tex_path, bib_path=bib_path,
                diagnostics=diagnostics,
            )
            log.write(f"Report written to {report_path}")
            if suggestions:
                # Drop the suggestion report next to the alignment report.
                sug_path = str(Path(report_path).with_name(
                    Path(report_path).stem + "_suggestions.md"
                ))
                sd.write_markdown_report(
                    suggestions, entries, sug_path,
                    threshold=suggestion_threshold,
                )
                log.write(f"Suggestion report written to {sug_path}")

        progress.progress(100, text="Done.")

        # ----- display -----
        # Status counts
        counts = {"ok": 0, "check": 0, "flag": 0, "no_data": 0}
        for r in results:
            s = r.get("status", "ok" if r.get("keyword_score") is not None else "no_data")
            counts[s] = counts.get(s, 0) + 1

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("OK", counts["ok"])
        col2.metric("Check", counts["check"])
        col3.metric("Flag", counts["flag"])
        col4.metric("No data", counts["no_data"])

        st.subheader(f"All citations ({len(results)})")
        rows = []
        for r in results:
            rows.append({
                "status": r.get("status", "ok" if r.get("keyword_score") is not None else "no_data"),
                "key": r["key"],
                "embed": r.get("embedding_score"),
                "tfidf": r.get("tfidf_score"),
                "nli_polarity": r.get("nli_polarity"),
                "nli_label": r.get("nli_label"),
                "llm": (r.get("llm_verdict") or {}).get("verdict") if r.get("llm_verdict") else None,
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

        def _render_citation_block(r):
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
                st.markdown(f"**LLM verdict**: {v.get('verdict')} — {v.get('reason')}")
            if r.get("best_snippet"):
                st.markdown(f"**Best-matching sentence in cited paper**: _{r['best_snippet']}_")
            st.markdown("**Citation context(s)**:")
            for ctx in r["contexts"][:3]:
                st.markdown(f"> {ctx[:400]}")
            st.markdown("---")

        flag_results = [r for r in results if r.get("status") == "flag"]
        check_results = [r for r in results if r.get("status") == "check"]
        if flag_results:
            with st.expander(f"FLAG ({len(flag_results)})", expanded=True):
                for r in flag_results:
                    _render_citation_block(r)
        if check_results:
            with st.expander(f"CHECK ({len(check_results)})", expanded=False):
                for r in check_results:
                    _render_citation_block(r)

        # ---- Citation suggestions ----
        if suggestions:
            st.subheader(
                f"Citation suggestions ({len(suggestions)} sentences "
                f"with score >= {suggestion_threshold:.2f})"
            )
            sug_rows = [
                {
                    "score": s["best_score"],
                    "file": Path(s["file"]).name,
                    "suggested_key": s["suggestions"][0]["key"],
                    "currently_cited": ", ".join(s.get("cite_keys") or []) or "—",
                    "sentence": s["raw"][:200].replace("\n", " "),
                }
                for s in sorted(suggestions, key=lambda x: -x["best_score"])
            ]
            st.dataframe(sug_rows, width="stretch", height=300)

            with st.expander("Per-sentence detail with top suggestions", expanded=False):
                for s in sorted(suggestions, key=lambda x: -x["best_score"])[:50]:
                    st.markdown(
                        f"**{Path(s['file']).name}** — score {s['best_score']:.3f}"
                    )
                    st.markdown(f"> {s['raw'][:500]}")
                    existing = s.get("cite_keys") or []
                    if existing:
                        st.markdown(f"_Already cites: {', '.join(existing)}_")
                    else:
                        st.markdown("_No citation in this sentence._")
                    for sug in s["suggestions"]:
                        title = entries.get(sug["key"], {}).get("title", "")
                        marker = " — *(already cited)*" if sug["key"] in existing else ""
                        st.markdown(f"- `{sug['key']}` ({sug['score']:.3f}) — {title}{marker}")
                    st.markdown("---")
        elif suggest_missing and mode != "PDF (via GROBID)":
            st.info(
                f"No sentences scored above the suggestion threshold "
                f"({suggestion_threshold:.2f}). Try lowering the slider."
            )
        elif suggest_missing and mode == "PDF (via GROBID)":
            st.warning("Citation suggestions are only available in LaTeX modes for now.")

        # ---- Downloads ----
        cols = st.columns(2)
        if report_path and Path(report_path).exists():
            cols[0].download_button(
                "Download alignment report",
                Path(report_path).read_text(encoding="utf-8"),
                file_name=Path(report_path).name,
                mime="text/markdown",
            )
        if suggestions and report_path:
            sug_path = Path(report_path).with_name(
                Path(report_path).stem + "_suggestions.md"
            )
            if sug_path.exists():
                cols[1].download_button(
                    "Download suggestions report",
                    sug_path.read_text(encoding="utf-8"),
                    file_name=sug_path.name,
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
