"""Thin HTTP client for a locally-running GROBID instance."""

from pathlib import Path
import requests


DEFAULT_URL = "http://localhost:8070"


def is_alive(base_url: str = DEFAULT_URL, timeout: float = 2.0) -> bool:
    try:
        r = requests.get(f"{base_url}/api/isalive", timeout=timeout)
        return r.status_code == 200 and r.text.strip().lower() == "true"
    except requests.RequestException:
        return False


def process_fulltext(
    pdf_path: str,
    base_url: str = DEFAULT_URL,
    consolidate_citations: int = 1,
    timeout: float = 600.0,
) -> str:
    """POST a PDF and return the TEI XML string.

    consolidate_citations:
        0 = no consolidation
        1 = lookup against CrossRef (sends titles only) — adds DOIs, fixes typos
        2 = lookup but only fill missing fields
    """
    url = f"{base_url}/api/processFulltextDocument"
    pdf = Path(pdf_path)
    with pdf.open("rb") as fh:
        files = {"input": (pdf.name, fh, "application/pdf")}
        data = {
            "consolidateCitations": str(consolidate_citations),
            "includeRawCitations": "1",
            "segmentSentences": "1",
        }
        r = requests.post(url, files=files, data=data, timeout=timeout)
    r.raise_for_status()
    return r.text
