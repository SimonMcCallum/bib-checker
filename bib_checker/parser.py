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


def parse_bib_dir(bib_dir: str) -> dict:
    """Merge every .bib file under a directory into one entries dict.

    On duplicate keys, the first-seen entry wins (later ones are ignored
    rather than silently overwriting earlier ones).
    """
    merged: dict = {}
    root = Path(bib_dir)
    for bib_file in sorted(root.rglob("*.bib")):
        for k, v in parse_bib(str(bib_file)).items():
            merged.setdefault(k, v)
    return merged


def load_bibs_smart(
    bib_arg: str,
    tex_arg: str | None = None,
    tex_only: bool = False,
) -> tuple[dict, str]:
    """Resolve a bib path to (merged_entries, primary_path).

    Default behaviour (``tex_only=False``): a single .bib file uses just that
    file; a directory merges *every* .bib found under it (recursively), with
    a field-level merge on duplicate keys — existing fields are kept,
    missing ones are filled from the next .bib that has them. .tex-referenced
    bibs are placed first in the merge order so their data takes precedence
    on collisions.

    Strict mode (``tex_only=True``): only the .bib files actually referenced
    via ``\\bibliography{}`` / ``\\addbibresource{}`` in the .tex source are
    loaded; other .bib files are ignored. Useful if you maintain a personal
    reference library alongside the project bib and want to avoid pulling
    in irrelevant entries.

    Lives here so the CLI and the GUI share the same logic.
    """
    p = Path(bib_arg)
    if p.is_file():
        return parse_bib(str(p)), str(p)
    if not p.is_dir():
        raise FileNotFoundError(f"{bib_arg} not found")

    tex_refs: list[Path] = find_bib_paths_from_tex(tex_arg) if tex_arg else []

    if tex_only:
        if not tex_refs:
            raise FileNotFoundError(
                f"--tex-bib-only set but no \\bibliography or \\addbibresource "
                f"references found in {tex_arg}"
            )
        ordered = tex_refs
    else:
        # All .bib files anywhere under the directory; tex-referenced ones
        # appear first so they take precedence on field-merge ties.
        seen: set = set()
        ordered = []
        for b in tex_refs:
            key = b.resolve()
            if key not in seen:
                seen.add(key)
                ordered.append(b)
        for b in sorted(p.rglob("*.bib")):
            key = b.resolve()
            if key not in seen:
                seen.add(key)
                ordered.append(b)
        if not ordered:
            raise FileNotFoundError(f"No .bib files found under {p}")

    merged: dict = {}
    for bib_file in ordered:
        for k, v in parse_bib(str(bib_file)).items():
            if k not in merged:
                merged[k] = dict(v)
            else:
                existing = merged[k]
                for fk, fv in v.items():
                    if fv and not existing.get(fk):
                        existing[fk] = fv
    primary = tex_refs[0] if tex_refs else ordered[0]
    return merged, str(primary)


def find_bib_paths_from_tex(
    tex_path: str,
    exclude_dirs: tuple = ("Figures", "figures"),
) -> list[Path]:
    r"""Discover the .bib files actually referenced by the LaTeX source.

    Scans every .tex file under tex_path for ``\bibliography{...}`` and
    ``\addbibresource{...}`` commands, then resolves each referenced name
    against the project tree. The same name may sit in a subfolder, so we
    rglob the project root for the basename.

    Returns a deduplicated list of resolved paths in the order they were
    encountered.
    """
    p = Path(tex_path)
    tex_root = p if p.is_dir() else p.parent
    if p.is_file():
        tex_files = [p]
    else:
        tex_files = [
            t for t in p.rglob("*.tex")
            if not any(part in exclude_dirs for part in t.relative_to(p).parts)
        ]

    requested: list[str] = []
    for tex_file in tex_files:
        try:
            text = tex_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Strip line-comments so commented-out bibliography commands are ignored.
        text = re.sub(r"(?<!\\)%.*$", "", text, flags=re.MULTILINE)
        for m in re.finditer(r"\\bibliography\s*\{([^}]+)\}", text):
            for name in m.group(1).split(","):
                if name.strip():
                    requested.append(name.strip())
        for m in re.finditer(r"\\addbibresource\s*\{([^}]+)\}", text):
            if m.group(1).strip():
                requested.append(m.group(1).strip())

    resolved: list[Path] = []
    seen: set = set()
    for name in requested:
        bib_name = name if name.lower().endswith(".bib") else f"{name}.bib"
        # First try relative-to-root, then a tree-wide search by basename.
        candidates = [tex_root / bib_name, *tex_root.rglob(bib_name)]
        for cand in candidates:
            try:
                key = cand.resolve()
            except OSError:
                continue
            if cand.is_file() and key not in seen:
                resolved.append(cand)
                seen.add(key)
                break
    return resolved


def extract_citations_from_dir(tex_dir: str, exclude_dirs: tuple = ("Figures", "figures")) -> dict:
    """Walk a directory for .tex files and merge their citation contexts.

    Skips paths under any folder named in `exclude_dirs` (TikZ figure files
    typically live in Figures/ and contain no citations worth merging).
    """
    citations: dict = {}
    root = Path(tex_dir)
    for tex_file in root.rglob("*.tex"):
        # Skip excluded subtrees
        if any(part in exclude_dirs for part in tex_file.relative_to(root).parts):
            continue
        for k, ctxs in extract_citations(str(tex_file)).items():
            citations.setdefault(k, []).extend(ctxs)
    return citations


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
