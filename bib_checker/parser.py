"""Parse .bib and .tex files."""

import re
from pathlib import Path


def parse_bib(bib_path: str) -> dict:
    """Parse a .bib file into a dict keyed by citation key.

    Returns dict of {key: {field: value, ...}} with entry_type included.
    Uses manual parsing to avoid bibtexparser version issues.
    """
    text = Path(bib_path).read_text(encoding="utf-8", errors="replace")
    entries = {}

    # Match @type{key, ... }
    pattern = re.compile(
        r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.+?)\n\}", re.DOTALL
    )
    for match in pattern.finditer(text):
        entry_type = match.group(1).lower()
        key = match.group(2).strip()
        body = match.group(3)

        fields = {"entry_type": entry_type}
        # Match field = {value} or field = "value"
        field_pattern = re.compile(
            r"(\w+)\s*=\s*[{\"](.+?)[}\"](?:\s*,|\s*$)", re.DOTALL
        )
        for fm in field_pattern.finditer(body):
            fname = fm.group(1).lower().strip()
            fval = fm.group(2).strip()
            # Clean up LaTeX braces
            fval = re.sub(r"[{}]", "", fval)
            fields[fname] = fval

        entries[key] = fields

    return entries


def write_bib_field(bib_path: str, key: str, field: str, value: str) -> bool:
    """Add or update a field in a bib entry. Returns True if modified."""
    text = Path(bib_path).read_text(encoding="utf-8", errors="replace")

    # Find the entry
    entry_pattern = re.compile(
        rf"(@\w+\{{\s*{re.escape(key)}\s*,.*?)\n\}}",
        re.DOTALL,
    )
    match = entry_pattern.search(text)
    if not match:
        return False

    entry_text = match.group(1)

    # Check if field already exists
    field_pattern = re.compile(
        rf"(\s*{re.escape(field)}\s*=\s*\{{)(.+?)(\}})",
        re.DOTALL,
    )
    field_match = field_pattern.search(entry_text)

    # Escape value for bib
    clean_value = value.replace("{", "").replace("}", "")

    if field_match:
        # Update existing field
        new_entry = entry_text[: field_match.start(2)] + clean_value + entry_text[field_match.end(2) :]
    else:
        # Add new field before the closing brace
        new_entry = entry_text + f"\n  {field} = {{{clean_value}}},"

    new_text = text[: match.start(1)] + new_entry + text[match.end(1) :]
    Path(bib_path).write_text(new_text, encoding="utf-8")
    return True


def extract_citations(tex_path: str) -> dict:
    """Extract citation contexts from a .tex file.

    Returns dict of {citation_key: [context_string, ...]}.
    Each context is the sentence or surrounding text containing the citation.
    """
    text = Path(tex_path).read_text(encoding="utf-8", errors="replace")

    # Remove comments
    text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)

    citations = {}

    # Match \cite{key}, \citep{key}, \citet{key}, \citeauthor{key}
    cite_pattern = re.compile(
        r"\\cite[pt]?\s*(?:\[[^\]]*\])?\s*\{([^}]+)\}"
    )

    for match in cite_pattern.finditer(text):
        keys_str = match.group(1)
        keys = [k.strip() for k in keys_str.split(",")]

        # Extract surrounding context (200 chars each side)
        start = max(0, match.start() - 200)
        end = min(len(text), match.end() + 200)
        context = text[start:end].strip()

        # Clean LaTeX commands for readability
        context = re.sub(r"\\[a-zA-Z]+\*?\s*", " ", context)
        context = re.sub(r"[{}$~]", " ", context)
        context = re.sub(r"\s+", " ", context).strip()

        for key in keys:
            if key not in citations:
                citations[key] = []
            citations[key].append(context)

    return citations
