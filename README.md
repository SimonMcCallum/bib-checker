# bib-checker

Verify LaTeX/PDF bibliography entries, fetch abstracts, check whether citations actually support the claims made in a paper, detect citations whose findings *contradict* the claim, and suggest citations from a personal bib library — all locally, without sending PDF content to a generative AI service.

## Install

The package is split into tiers so you only install what you need:

```bash
# Core: .tex + .bib, verify, fetch abstracts, TF-IDF alignment, suggest, LLM (HTTP)
pip install -e .

# Add semantic similarity (sentence-transformers, ~500MB inc. torch)
pip install -e ".[embeddings]"

# Add local NLI claim-inversion scorer (same model family as embeddings)
pip install -e ".[nli]"

# Add PDF input via GROBID (lxml + a running GROBID Docker container)
pip install -e ".[pdf]"

# Add the browser GUI
pip install -e ".[gui]"

# Everything except FAISS (most users don't need it)
pip install -e ".[all]"
```

First run of any `embedding`-mode command downloads `sentence-transformers/all-MiniLM-L6-v2` (~80MB) to `~/.cache/huggingface`. The NLI scorer additionally downloads `cross-encoder/nli-deberta-v3-base` (~440MB) on first use.

### GROBID (only needed for PDF input)

GROBID extracts structured references and inline citation markers from PDFs. It runs locally as a Docker container.

```powershell
# Windows
.\scripts\install-grobid.ps1   # docker pull (~500MB, one-time)
.\scripts\start-grobid.ps1     # start on http://localhost:8070
.\scripts\stop-grobid.ps1
```

```bash
# macOS / Linux
./scripts/install-grobid.sh
./scripts/start-grobid.sh
./scripts/stop-grobid.sh
```

GROBID's `consolidateCitations` mode is on by default — it sends *titles only* to CrossRef to enrich missing DOIs and fix typos. No PDF content is shared.

## Usage

```bash
# --- LaTeX input ---

# Verify all bib entries exist in CrossRef/Semantic Scholar
bib-checker verify references.bib

# Fetch abstracts and write them into the bib file
bib-checker abstracts references.bib

# Check citation alignment from a .tex source (embedding scorer by default)
bib-checker check paper.tex references.bib

# Or pass a directory — bib-checker will recursively scan all .tex files
# (Figures/ subfolders are excluded automatically). Useful for thesis projects
# that split chapters into multiple .tex files.
bib-checker check ./thesis-source/ references.bib

# Add the local NLI claim-inversion check on top of embedding similarity
bib-checker check paper.tex references.bib --check-polarity

# Generate a full markdown report
bib-checker report paper.tex references.bib -o report.md

# --- PDF input (requires GROBID running) ---

# Extract a .bib file from a PDF
bib-checker pdf-extract paper.pdf -o references.bib

# Full pipeline: extract bib, fetch abstracts, score alignment, write report
bib-checker pdf-check paper.pdf -o report.md

# --- Citation suggestions (RAG over your personal library) ---

# Build/cache embeddings from your bib and return top-5 candidate citations for a passage
bib-checker suggest mylib.bib --query "Recent work on transformer attention sparsity..."

# Or read the query from a file (e.g. the paragraph you're writing)
bib-checker suggest mylib.bib --query-file draft_paragraph.txt -k 10
```

## End-to-end example: a thesis folder

Run the whole pipeline against a multi-chapter LaTeX thesis, write an augmented bib (original `.bib` + fetched abstracts + Semantic Scholar tldrs) next to the original, and drop a markdown report in the same folder:

```powershell
python -m bib_checker.cli report `
  "G:\git\thesiscode\Thesis" `
  "G:\git\thesiscode\Thesis\thesis.bib" `
  --bib-out "G:\git\thesiscode\Thesis\thesis_with_abstracts.bib" `
  -o      "G:\git\thesiscode\Thesis\alignment_report.md" `
  --fetch-missing
```

Walks every `.tex` under the folder (`Figures/` excluded), copies the bib so the original is untouched, fetches abstracts and tldrs **only for entries actually cited** in the .tex (skips the rest of a personal library), then runs sentence-level embedding alignment.

Optional knobs you can stack on:

```powershell
# Add the local NLI claim-inversion check (~440 MB model on first run)
  --check-polarity

# Lighter NLI model (~80 MB)
  --check-polarity --nli-model cross-encoder/nli-MiniLM2-L6-H768

# Add a local-LLM second pass via Ollama (no cloud)
  --llm ollama --llm-model llama3.1

# Or a cloud LLM (set $env:ANTHROPIC_API_KEY first)
  --llm anthropic --llm-model claude-haiku-4-5-20251001

# Switch from semantic embeddings to lexical TF-IDF (no model needed)
  --scorer tfidf

# Also fetch entries that aren't cited (slower; populates the whole library)
  --fetch-missing --fetch-all

# Drop the threshold to catch borderline cases (default 0.30)
  --threshold 0.25
```

### Same run from the GUI

```powershell
pip install -e ".[gui,embeddings,nli]"
bib-checker-gui   # opens http://localhost:8501 in your browser
```

In the sidebar:

| Field | Value |
|---|---|
| **Source** | LaTeX folder (recursive) |
| **Path to LaTeX folder** | `G:\git\thesiscode\Thesis` |
| **Path to .bib file** | `G:\git\thesiscode\Thesis\thesis.bib` *(auto-suggested)* |
| **Output report (.md)** | `G:\git\thesiscode\Thesis\alignment_report.md` |
| **Similarity scorer** | `embedding` |
| **Flag below similarity** | `0.30` |
| **Fetch missing abstracts** | ☑ |
| **Run NLI claim-inversion check** | ☐ (toggle on for polarity flips) |
| **LLM second-pass review** | `none` (or pick one) |

Click **Run check**. The progress bar streams each phase (extract → fetch → score → NLI → LLM → report). When done, the table of all citations renders in the main pane, flagged ones are expandable below it, and there's a **Download markdown report** button at the bottom.

> The GUI doesn't yet expose `--bib-out`. To preserve `thesis.bib`, copy it to `thesis_with_abstracts.bib` first and point the GUI at the copy.

## Browser GUI

After `pip install -e ".[gui]"`:

```bash
bib-checker-gui
```

This opens a Streamlit page in your browser where you can:

- Pick the input mode (single .tex, .tex folder, or PDF).
- Browse to the bib file (suggested next to the input).
- Toggle scorer (embedding vs TF-IDF), abstract fetch, NLI polarity check, and LLM second-pass via checkboxes.
- See the alignment table, flagged citations with the best-matching cited sentence, and download the markdown report.

## Claim-inversion (polarity) check

The classical failure mode is citing a paper whose findings *contradict* the claim you're making — embedding similarity scores high (the topic matches) but the polarity is flipped. Pass `--check-polarity` to add a local NLI cross-encoder pass that scores each citation pair as entailment / neutral / contradiction. A high `nli_polarity` (= contradiction − entailment) raises a flag the embedding scorer would have missed.

```bash
bib-checker check paper.tex references.bib --check-polarity
# Lighter model if disk/RAM constrained:
bib-checker check paper.tex references.bib --check-polarity \
  --nli-model cross-encoder/nli-MiniLM2-L6-H768
```

## Optional LLM review

By default, `bib-checker` is fully algorithmic and **no PDF content leaves your machine**. If you want a second-pass review of flagged citations, you can opt into one:

```bash
# Local model via Ollama (no content leaves your machine)
ollama pull llama3.1
bib-checker pdf-check paper.pdf --llm ollama --llm-model llama3.1

# Anthropic Claude (cloud — sends context + title + abstract)
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell
bib-checker pdf-check paper.pdf --llm anthropic
# or pick a stronger model:
bib-checker pdf-check paper.pdf --llm anthropic --llm-model claude-sonnet-4-6

# OpenAI (cloud)
$env:OPENAI_API_KEY = "sk-..."
bib-checker pdf-check paper.pdf --llm openai --llm-model gpt-4o-mini

# Any OpenAI-compatible endpoint (e.g. self-hosted vLLM, Together, etc.)
bib-checker pdf-check paper.pdf --llm openai --llm-base-url https://my-server.example/v1

# Send EVERY citation to the LLM, not just flagged ones (slower/costlier)
bib-checker pdf-check paper.pdf --llm anthropic --llm-all
```

The LLM returns a verdict (`support` / `tangential` / `mismatch` / `unknown`) plus a one-sentence reason for each citation reviewed; verdicts of `mismatch` or `tangential` will also flag the citation in the report even if the embedding score was above threshold (this catches the polarity-flip case where the cited paper *contradicts* the claim it's used to support).

## How it works

1. **verify** — Searches CrossRef and Semantic Scholar by title/DOI for each entry. Reports found/not-found, year/author match.
2. **abstracts** — Fetches abstracts from CrossRef/Semantic Scholar and writes them into the bib file as `abstract = {...}` fields.
3. **pdf-extract** — Sends the PDF to GROBID, parses the TEI XML response into BibTeX entries plus an inline-citation map (each `\ref` already linked to its `<biblStruct>`).
4. **check / pdf-check** — For each cited entry, encodes the citation context and the abstract (or title fallback) using a local sentence-transformer, then computes cosine similarity. Flags citations below `--threshold` (default 0.30 for embeddings, 0.08 for TF-IDF).
5. **suggest** — Encodes every entry in the user's bib once (cached on disk by mtime), then returns the top-k most similar entries to a query passage. Use this when drafting to find candidate citations from things you've already read.

## Scoring choices

- **Embedding (default)**: `all-MiniLM-L6-v2` cosine similarity. Captures semantic similarity, e.g. "we use a transformer model" matches an abstract about "self-attention architectures". ~5× slower than TF-IDF on first run (model load), then microseconds per pair.
- **TF-IDF (`--scorer tfidf`)**: faster, no model needed, but only catches lexical overlap.

## When to use FAISS

For the alignment task on a single paper (~50–300 references), `bib-checker` uses a plain numpy dot product — at that scale FAISS adds no useful speedup. The `suggest` command is the same: numpy stays fast through ~10,000 entries (~5ms per query). Past that, set `embed.FAISS_THRESHOLD` lower or `pip install faiss-cpu` and the code switches automatically.

## Privacy

| Step | Data sent over the network |
|------|----------------------------|
| `verify`, `abstracts` | Title + author + year (and DOI when known) to CrossRef + Semantic Scholar |
| `pdf-extract` (default) | PDF sent to **localhost** GROBID; GROBID then sends *titles only* to CrossRef for consolidation. Pass `--no-consolidate` (TODO) or run with `consolidateCitations=0` to disable. |
| `check`, `pdf-check`, `suggest` | Nothing — embeddings run locally |
