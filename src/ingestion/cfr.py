"""
Phase 3 — Code of Federal Regulations (CFR) Ingestion

Downloads selected CFR titles from the free eCFR API (ecfr.gov),
parses the XML, chunks with overlap, and writes to
data/processed/us_regulations.jsonl

Selected titles:
    12 — Banks and Banking
    21 — Food and Drugs (FDA)
    26 — Internal Revenue (Tax regulations)
    29 — Labor (OSHA, worker rights)
    40 — Protection of Environment (EPA)
    45 — Public Welfare (HHS)
    49 — Transportation

No API key required. Completely free.

Run:
    python -m src.ingestion.cfr

Output:
    data/raw/cfr/title_{n}.xml           — cached raw XML per title
    data/processed/us_regulations.jsonl  — chunked + tagged regulations
"""

import re
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path

import jsonlines
import requests
from tqdm import tqdm

from src.ingestion.chunker import chunk_text

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR  = Path("data/raw/cfr")
OUT_FILE = Path("data/processed/us_regulations.jsonl")

ECFR_BASE  = "https://www.ecfr.gov/api/versioner/v1"
TODAY      = date.today().isoformat()   # e.g. "2026-04-25"
SLEEP_SEC  = 1.2                        # between requests

SELECTED_TITLES = {
    12: "Banks and Banking",
    21: "Food and Drugs",
    26: "Internal Revenue",
    29: "Labor",
    40: "Protection of Environment",
    45: "Public Welfare",
    49: "Transportation",
}

# XML namespaces used by eCFR
NS = {
    "": "https://www.ecfr.gov/current/",
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get_bytes(url: str, retries: int = 5) -> bytes:
    """GET raw bytes with exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code == 429:
                wait = 2 ** attempt * 30
                tqdm.write(f"  [RATE LIMIT] Waiting {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = 2 ** attempt * 5
                tqdm.write(f"  [SERVER ERROR {resp.status_code}] Waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            time.sleep(SLEEP_SEC)
            return resp.content
        except requests.RequestException as e:
            wait = 2 ** attempt * 5
            tqdm.write(f"  [ERROR] {e} — retrying in {wait}s (attempt {attempt + 1}/{retries})")
            time.sleep(wait)
    return b""

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_title(title_num: int) -> Path:
    """
    Download a full CFR title XML and cache it locally.
    Returns path to the cached file.
    """
    raw_path = RAW_DIR / f"title_{title_num:02d}.xml"
    if raw_path.exists() and raw_path.stat().st_size > 1000:
        tqdm.write(f"  [CACHED] Title {title_num}")
        return raw_path

    url = f"{ECFR_BASE}/full/{TODAY}/title-{title_num}.xml"
    tqdm.write(f"  [DOWNLOAD] Title {title_num} from {url}")
    content = _get_bytes(url)
    if content:
        raw_path.write_bytes(content)
    return raw_path

# ---------------------------------------------------------------------------
# XML parsing helpers
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    """Strip extra whitespace from extracted text."""
    return re.sub(r"\s+", " ", text or "").strip()


def _iter_sections(root: ET.Element):
    """
    Walk the XML tree and yield (part, section_num, heading, full_text) tuples.
    eCFR XML uses tags like <DIV5> for parts, <DIV8> for sections.
    Tries both namespaced and bare tags for robustness.
    """
    # eCFR XML tags: PART = DIV5, SUBPART = DIV6, SECTION = DIV8
    # Also tries lowercase variants
    section_tags = {"SECTION", "section", "DIV8", "div8"}
    part_tags    = {"PART", "part", "DIV5", "div5"}

    current_part = ""

    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag  # strip namespace

        if tag in part_tags:
            # Extract part number/heading
            head = elem.find("HEAD") or elem.find("head")
            current_part = _clean(head.text) if head is not None else ""

        if tag in section_tags:
            # Extract section number
            sectno = elem.find("SECTNO") or elem.find("sectno")
            section_num = _clean(sectno.text) if sectno is not None else ""

            # Extract heading
            subject = elem.find("SUBJECT") or elem.find("subject")
            heading = _clean(subject.text) if subject is not None else ""

            # Extract all paragraph text
            paragraphs = []
            for child in elem.iter():
                child_tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
                if child_tag in {"P", "p", "FP", "fp", "EXTRACT", "extract"}:
                    if child.text:
                        paragraphs.append(_clean(child.text))

            full_text = f"{section_num} {heading}. " + " ".join(paragraphs)
            full_text = _clean(full_text)

            if len(full_text) > 80:
                yield current_part, section_num, heading, full_text


def _parse_title(xml_path: Path, title_num: int, title_name: str) -> list[dict]:
    """Parse one CFR title XML file into a list of chunks."""
    if not xml_path.exists() or xml_path.stat().st_size < 100:
        return []

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        tqdm.write(f"  [PARSE ERROR] Title {title_num}: {e}")
        return []

    chunks = []
    for part, section_num, heading, text in _iter_sections(root):
        # Build citation: e.g. "29 CFR § 1910.132"
        sec_clean = section_num.replace("§", "").strip()
        citation  = f"{title_num} CFR § {sec_clean}" if sec_clean else f"{title_num} CFR"

        metadata = {
            "source":       "cfr",
            "jurisdiction": "us",
            "collection":   "us_regulations",
            "title_num":    title_num,
            "title_name":   title_name,
            "part":         part,
            "section_num":  section_num,
            "heading":      heading,
            "citation":     citation,
        }
        chunks.extend(chunk_text(text, metadata))

    return chunks

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def ingest() -> int:
    """
    Download and chunk all selected CFR titles into data/processed/us_regulations.jsonl.
    Skips titles already cached in data/raw/cfr/.

    Returns:
        total number of chunks written
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    total_chunks = 0

    with jsonlines.open(OUT_FILE, mode="w") as writer:
        for title_num, title_name in tqdm(SELECTED_TITLES.items(), desc="CFR titles", unit="title"):
            tqdm.write(f"\nProcessing Title {title_num} — {title_name}")

            xml_path = _download_title(title_num)
            chunks   = _parse_title(xml_path, title_num, title_name)

            for chunk in chunks:
                writer.write(chunk)

            total_chunks += len(chunks)
            tqdm.write(f"  → {len(chunks):,} chunks from Title {title_num}")

    print(f"\nDone. {total_chunks:,} total chunks written to {OUT_FILE}")
    return total_chunks


if __name__ == "__main__":
    ingest()
