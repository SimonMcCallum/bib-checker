# bib-checker

Verify LaTeX bibliography entries, fetch abstracts, and check whether citations actually support the claims made in a paper.

## Install

```bash
pip install -e .
```

## Usage

```bash
# Verify all bib entries exist in CrossRef/Semantic Scholar
bib-checker verify references.bib

# Fetch abstracts and add them to bib entries
bib-checker abstracts references.bib

# Check citation-abstract alignment in a paper
bib-checker check paper.tex references.bib

# Generate a full markdown report
bib-checker report paper.tex references.bib -o report.md
```

## How it works

1. **verify** — Searches CrossRef and Semantic Scholar by title/DOI for each entry. Reports found/not-found, year match, author match.
2. **abstracts** — Fetches abstracts from Semantic Scholar and writes them into the bib file as `abstract = {...}` fields.
3. **check** — Parses `\cite{}` commands in .tex, extracts surrounding context, compares against abstracts using TF-IDF cosine similarity.
4. **report** — Combines all of the above into a markdown report with alignment scores and flags.
