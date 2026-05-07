"""Parse a PDF via GROBID into bib entries + citation contexts.

Returns the same shapes as parser.parse_bib / parser.extract_citations so the
rest of the pipeline (verify, abstracts, alignment) doesn't need to change.
"""

from pathlib import Path
import re
from rich.console import Console

try:
    from lxml import etree
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "lxml is required for PDF input via GROBID. "
        "Install it with: pip install bib-checker[pdf]"
    ) from _e

from . import grobid_client

console = Console()

NS = {"tei": "http://www.tei-c.org/ns/1.0"}

# Titles that indicate a TEI <biblStruct> was parsed from a chapter heading,
# appendix header, or thesis section rather than a real reference. GROBID
# occasionally synthesises one of these when a thesis has odd back-matter, and
# its CrossRef-consolidation step can then match the heading to an unrelated
# real paper, so the entry looks legitimate at a glance.
_JUNK_PREFIX = re.compile(
    r"^\s*(appendix|chapter\s+\d+|bibliograph(y|ies)|references?|index|"
    r"contents?|acknowledg(e?ments?)?|foreword|preface|"
    r"declaration|dedication|copyright|table\s+of\s+contents)\b",
    re.IGNORECASE,
)
# Standalone thesis-section headings — match only short whole titles to avoid
# clobbering real references that happen to start with these words.
_JUNK_SECTION = {
    "abstract", "introduction", "background", "literature review",
    "related work", "methods", "methodology", "results", "findings",
    "discussion", "conclusion", "conclusions", "future work",
    "limitations", "recommendations", "research aim", "research aims",
    "research aims and objectives", "data analysis", "data collection",
    "hypotheses", "summary", "overview",
}


_YEAR_RE = re.compile(r"\b(19[0-9]{2}|20[0-2][0-9])\b")
_SURNAME_RE = re.compile(r"\b([A-Z][A-Za-z'À-ſ-]{1,})\b")
# Strip Latin abbreviations and connectors that are not surnames.
_NON_SURNAME = re.compile(
    r"\b(et\s*al\.?|and|von|van|de|der|du|le|la|the|al|in|et)\b",
    re.IGNORECASE,
)


def _candidate_surnames(text: str) -> list[str]:
    """Pull capitalised tokens that could be surnames from a citation marker.

    Handles compact forms produced by some PDFs:
        'Makranskyetal.,2021' -> ['makransky']
        '(MacDonald, Kreutz, & Mitchell, 2012)' -> ['macdonald','kreutz','mitchell']
        'introduced by Paul Dourish (2001)' -> ['paul','dourish']
    """
    if not text:
        return []
    # Split jammed "Surnameetal" -> "Surname"
    text = re.sub(r"([A-Za-z]+)etal\b", r"\1", text)
    # Drop connectors so they don't get picked up as surnames
    text = _NON_SURNAME.sub(" ", text)
    return [m.lower() for m in _SURNAME_RE.findall(text)]


def _entry_surnames(entry: dict) -> list[str]:
    """Surnames of all authors in a bib entry, in order."""
    author = entry.get("author", "") or ""
    out = []
    for chunk in author.split(" and "):
        parts = chunk.strip().split()
        if parts:
            out.append(parts[-1].lower())
    return out


def _try_fuzzy_link(year: str, candidates: list[str], entries: dict) -> str | None:
    """Match an inline citation to a bib entry using surname + year.

    Returns a key only if exactly one entry matches.
    """
    if not year or not candidates:
        return None
    cand_set = set(candidates)
    matches = []
    for key, entry in entries.items():
        if (entry.get("year") or "") != year:
            continue
        surnames = _entry_surnames(entry)
        if not surnames:
            continue
        first = surnames[0]
        if first not in cand_set:
            continue
        # If the bib entry AND the inline both have multiple authors, require at
        # least one secondary surname overlap — this prevents linking different
        # papers that share a first author + year.
        if len(surnames) > 1 and len(cand_set) > 1:
            secondary_entry = set(surnames[1:])
            secondary_inline = cand_set - {first}
            if secondary_inline and secondary_entry and not (secondary_entry & secondary_inline):
                continue
        matches.append(key)
    return matches[0] if len(matches) == 1 else None


def _is_junk_title(title: str) -> bool:
    if not title:
        return False
    t = title.strip()
    if _JUNK_PREFIX.match(t):
        return True
    # Exact-match thesis section headings, case-insensitive, short titles only
    if len(t) <= 60 and t.lower().rstrip(".:") in _JUNK_SECTION:
        return True
    return False


def _text(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def _make_key(authors: list[str], year: str, title: str, used: set) -> str:
    """Build a citation key like Smith2020 or Smith2020a."""
    first_author = next((a for a in authors if a and a.strip()), "")
    parts = first_author.split()
    last = (parts[-1] if parts else "anon").lower()
    last = re.sub(r"[^a-z]", "", last) or "anon"
    base = f"{last.capitalize()}{year or 'nd'}"
    key = base
    suffix = "a"
    while key in used:
        key = f"{base}{suffix}"
        suffix = chr(ord(suffix) + 1)
    used.add(key)
    return key


def _bib_struct_to_entry(bs) -> dict:
    """Convert one <biblStruct> element to a dict matching parser.parse_bib output."""
    entry = {"entry_type": "article"}

    # Title — analytic preferred (article title), fall back to monogr (book/proc title)
    title_a = bs.find(".//tei:analytic/tei:title", NS)
    title_m = bs.find(".//tei:monogr/tei:title", NS)
    if title_a is not None and _text(title_a):
        entry["title"] = _text(title_a)
        if title_m is not None and _text(title_m):
            entry["journal"] = _text(title_m)
    elif title_m is not None and _text(title_m):
        entry["title"] = _text(title_m)
        entry["entry_type"] = "book"

    # Authors
    authors = []
    for pers in bs.findall(".//tei:analytic/tei:author/tei:persName", NS) or \
                bs.findall(".//tei:monogr/tei:author/tei:persName", NS):
        forenames = [_text(f) for f in pers.findall("tei:forename", NS)]
        surname = _text(pers.find("tei:surname", NS))
        full = " ".join([n for n in forenames + [surname] if n])
        if full:
            authors.append(full)
    if authors:
        entry["author"] = " and ".join(authors)

    # Year
    date_el = bs.find(".//tei:imprint/tei:date", NS)
    if date_el is not None:
        when = date_el.get("when") or _text(date_el)
        m = re.search(r"\b(19|20)\d{2}\b", when or "")
        if m:
            entry["year"] = m.group(0)

    # Volume / pages
    vol = bs.find(".//tei:imprint/tei:biblScope[@unit='volume']", NS)
    if vol is not None and _text(vol):
        entry["volume"] = _text(vol)
    pages = bs.find(".//tei:imprint/tei:biblScope[@unit='page']", NS)
    if pages is not None:
        from_p, to_p = pages.get("from"), pages.get("to")
        if from_p and to_p:
            entry["pages"] = f"{from_p}--{to_p}"
        elif _text(pages):
            entry["pages"] = _text(pages)

    # DOI
    doi = bs.find(".//tei:idno[@type='DOI']", NS)
    if doi is not None and _text(doi):
        entry["doi"] = _text(doi)

    # Publisher
    publisher = bs.find(".//tei:imprint/tei:publisher", NS)
    if publisher is not None and _text(publisher):
        entry["publisher"] = _text(publisher)

    # Raw citation as fallback
    raw = bs.find("tei:note[@type='raw_reference']", NS)
    if raw is not None and _text(raw):
        entry["_raw"] = _text(raw)

    return entry


def _extract_context(p_el, ref_el) -> str:
    """Extract sentence-level context around a <ref> element inside a <p>."""
    parent_text = "".join(p_el.itertext())
    target = "".join(ref_el.itertext())
    idx = parent_text.find(target) if target else -1
    if idx < 0:
        return parent_text.strip()[:400]
    # Sentence-ish window: 200 chars each side
    start = max(0, idx - 250)
    end = min(len(parent_text), idx + len(target) + 250)
    snippet = parent_text[start:end]
    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet


def parse_tei(tei_xml: str) -> tuple[dict, dict, dict, dict]:
    """Parse a GROBID TEI document.

    Returns:
        (entries, citations, xmlid_to_key, diagnostics)
        entries: {citation_key: {field: value, ...}}  — same shape as parser.parse_bib
        citations: {citation_key: [context_str, ...]} — same shape as parser.extract_citations
        xmlid_to_key: {tei_xml_id: citation_key}      — for debugging / cross-ref
        diagnostics: {
            "junk_entries":   [{"key", "title"}],         # bib entries that look like section headings
            "unused_entries": [{"key", "title"}],         # in bibliography, never cited inline
            "unresolved_inline": [{"text", "context"}],   # <ref> with no target / unknown target
        }
    """
    root = etree.fromstring(tei_xml.encode("utf-8"))

    # Bibliography pass 1: build entries and xmlid map
    entries: dict = {}
    xmlid_to_key: dict = {}
    junk_entries: list = []
    used_keys: set = set()
    for bs in root.findall(".//tei:listBibl/tei:biblStruct", NS):
        xml_id = bs.get("{http://www.w3.org/XML/1998/namespace}id") or ""
        entry = _bib_struct_to_entry(bs)
        title = entry.get("title", "") or ""

        if _is_junk_title(title):
            # Still allocate a key so we can report it, but keep it out of the
            # main entries dict so alignment / abstract fetching skip it.
            authors_list = (entry.get("author", "") or "").split(" and ")
            jkey = _make_key(authors_list, entry.get("year", ""), title, used_keys)
            junk_entries.append({"key": jkey, "title": title, "xml_id": xml_id})
            if xml_id:
                xmlid_to_key[xml_id] = None  # mark as junk so refs don't promote it
            continue

        authors_list = (entry.get("author", "") or "").split(" and ")
        key = _make_key(authors_list, entry.get("year", ""), title, used_keys)
        entries[key] = entry
        if xml_id:
            xmlid_to_key[xml_id] = key

    # Inline citations: every <ref type="bibr"> in the body
    citations: dict = {}
    cited_keys: set = set()
    unresolved_inline: list = []
    fuzzy_resolved = 0
    for ref in root.findall(".//tei:body//tei:ref[@type='bibr']", NS):
        target = ref.get("target", "") or ""
        ref_text = _text(ref) or ""

        # Walk up to enclosing <p> for context
        p_el = ref
        for _ in range(5):
            p_el = p_el.getparent()
            if p_el is None:
                break
            tag = etree.QName(p_el).localname
            if tag == "p":
                break
        ctx = _extract_context(p_el, ref) if p_el is not None else ref_text

        # 1. Trust GROBID's own linkage when it resolved
        if target.startswith("#"):
            xml_id = target[1:]
            key = xmlid_to_key.get(xml_id)
            if key:
                citations.setdefault(key, []).append(ctx)
                cited_keys.add(key)
                continue

        # 2. Fuzzy fallback: surname + year against the entries dict.
        # Pick up text immediately preceding the ref so narrative citations
        # like "Paul Dourish (2001)" still match.
        surname_text = ref_text
        if p_el is not None:
            parent_text = "".join(p_el.itertext())
            target_text = "".join(ref.itertext())
            idx = parent_text.find(target_text) if target_text else -1
            if idx > 0:
                surname_text = parent_text[max(0, idx - 60):idx] + " " + ref_text

        year_m = _YEAR_RE.search(ref_text) or _YEAR_RE.search(surname_text)
        year = year_m.group(0) if year_m else ""
        candidates = _candidate_surnames(surname_text)
        matched = _try_fuzzy_link(year, candidates, entries)
        if matched:
            citations.setdefault(matched, []).append(ctx)
            cited_keys.add(matched)
            fuzzy_resolved += 1
            continue

        # Filter false-positive <ref>s like "(8.15)" or "(M=36.15)" that GROBID
        # mistakenly tagged — drop if the marker text itself has no letters,
        # since real citations always contain a name or institution.
        if not re.search(r"[A-Za-z]{2,}", ref_text):
            continue
        unresolved_inline.append({"text": ref_text, "context": ctx[:300]})

    unused_entries = [
        {"key": k, "title": v.get("title", "")}
        for k, v in entries.items()
        if k not in cited_keys
    ]

    diagnostics = {
        "junk_entries": junk_entries,
        "unused_entries": unused_entries,
        "unresolved_inline": unresolved_inline,
        "fuzzy_resolved": fuzzy_resolved,
    }

    return entries, citations, xmlid_to_key, diagnostics


def write_bib(entries: dict, out_path: str) -> None:
    """Write entries dict to a .bib file."""
    lines = []
    for key, entry in entries.items():
        etype = entry.get("entry_type", "article")
        lines.append(f"@{etype}{{{key},")
        for field in ("author", "title", "journal", "year", "volume", "pages", "publisher", "doi"):
            val = entry.get(field)
            if val:
                clean = val.replace("{", "").replace("}", "")
                lines.append(f"  {field} = {{{clean}}},")
        lines.append("}\n")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def extract_from_pdf(
    pdf_path: str,
    base_url: str = grobid_client.DEFAULT_URL,
    tei_out: str | None = None,
) -> tuple[dict, dict, dict]:
    """Run a PDF through GROBID and return (entries, citations, diagnostics).

    diagnostics keys: junk_entries, unused_entries, unresolved_inline.
    If tei_out is given, also save the raw TEI XML for inspection.
    """
    if not grobid_client.is_alive(base_url):
        raise RuntimeError(
            f"GROBID is not reachable at {base_url}. "
            f"Start it with scripts/start-grobid.ps1 (Windows) or scripts/start-grobid.sh."
        )

    console.print(f"[dim]Sending {pdf_path} to GROBID...[/dim]")
    tei = grobid_client.process_fulltext(pdf_path, base_url=base_url)
    if tei_out:
        Path(tei_out).write_text(tei, encoding="utf-8")
        console.print(f"[dim]TEI saved to {tei_out}[/dim]")
    entries, citations, _, diagnostics = parse_tei(tei)
    console.print(
        f"[green]Extracted {len(entries)} bib entries and "
        f"{sum(len(v) for v in citations.values())} inline citations "
        f"({len(citations)} unique keys).[/green]"
    )
    if diagnostics.get("fuzzy_resolved"):
        console.print(
            f"[dim]Fuzzy-matched {diagnostics['fuzzy_resolved']} inline citations "
            f"that GROBID failed to link (surname + year fallback).[/dim]"
        )
    if diagnostics["junk_entries"]:
        console.print(
            f"[yellow]Filtered {len(diagnostics['junk_entries'])} junk entries "
            f"(section headings mis-parsed as references).[/yellow]"
        )
    if diagnostics["unused_entries"]:
        console.print(
            f"[yellow]{len(diagnostics['unused_entries'])} bib entries are "
            f"never cited inline.[/yellow]"
        )
    if diagnostics["unresolved_inline"]:
        console.print(
            f"[yellow]{len(diagnostics['unresolved_inline'])} inline citations "
            f"could not be resolved to a bib entry (cited but missing from references).[/yellow]"
        )
    return entries, citations, diagnostics
