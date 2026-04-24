"""
Phase 1 — US Code Ingestion

Downloads all 54 titles of the US Code as ZIP/XML from uscode.house.gov,
parses every section using the USLM XML schema, chunks with overlap,
and writes tagged records to data/processed/us_statutes.jsonl

Run:
    python -m src.ingestion.us_code

Output:
    data/raw/us_code/        — 54 ZIP files (kept for re-runs)
    data/processed/us_statutes.jsonl — ~600k chunks, ~180MB
"""

import io
import os
import re
import zipfile
from pathlib import Path

import jsonlines
import requests
from lxml import etree
from tqdm import tqdm

from src.ingestion.chunker import chunk_text

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw/us_code")
OUT_FILE = Path("data/processed/us_statutes.jsonl")

# ---------------------------------------------------------------------------
# Download URLs
# uscode.house.gov publishes each title as a ZIP containing XML.
# Pattern: /download/releasepoints/us/pl/118/usc{NN:02d}.zip
# Title 26 (IRC) is split into multiple appendix files — handled below.
# ---------------------------------------------------------------------------
BASE_URL = "https://uscode.house.gov/download/releasepoints/us/pl/118"

TITLE_NAMES: dict[int, str | None] = {
    1:  "General Provisions",
    2:  "The Congress",
    3:  "The President",
    4:  "Flag and Seal, Seat of Government, and the States",
    5:  "Government Organization and Employees",
    6:  "Domestic Security",
    7:  "Agriculture",
    8:  "Aliens and Nationality",
    9:  "Arbitration",
    10: "Armed Forces",
    11: "Bankruptcy",
    12: "Banks and Banking",
    13: "Census",
    14: "Coast Guard",
    15: "Commerce and Trade",
    16: "Conservation",
    17: "Copyrights",
    18: "Crimes and Criminal Procedure",
    19: "Customs Duties",
    20: "Education",
    21: "Food and Drugs",
    22: "Foreign Relations and Intercourse",
    23: "Highways",
    24: "Hospitals and Asylums",
    25: "Indians",
    26: "Internal Revenue Code",
    27: "Intoxicating Liquors",
    28: "Judiciary and Judicial Procedure",
    29: "Labor",
    30: "Mineral Lands and Mining",
    31: "Money and Finance",
    32: "National Guard",
    33: "Navigation and Navigable Waters",
    34: "Crime Control and Law Enforcement",
    35: "Patents",
    36: "Patriotic and National Observances, Ceremonies, and Organizations",
    37: "Pay and Allowances of the Uniformed Services",
    38: "Veterans Benefits",
    39: "Postal Service",
    40: "Public Buildings, Property, and Works",
    41: "Public Contracts",
    42: "The Public Health and Welfare",
    43: "Public Lands",
    44: "Public Printing and Documents",
    45: "Railroads",
    46: "Shipping",
    47: "Telecommunications",
    48: "Territories and Insular Possessions",
    49: "Transportation",
    50: "War and National Defense",
    51: "National and Commercial Space Programs",
    52: "Voting and Elections",
    53: None,  # Reserved — not enacted
    54: "National Park Service and Related Programs",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _download_zip(title_num: int, raw_dir: Path) -> Path | None:
    """Download the ZIP for a given title. Returns path to saved ZIP or None if title is reserved."""
    if TITLE_NAMES.get(title_num) is None:
        return None  # Title 53 is reserved

    zip_name = f"usc{title_num:02d}.zip"
    zip_path = raw_dir / zip_name

    if zip_path.exists():
        return zip_path  # Already downloaded

    url = f"{BASE_URL}/{zip_name}"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)
        return zip_path
    except requests.HTTPError as e:
        print(f"  [WARN] Title {title_num:02d} — HTTP {e.response.status_code}, skipping.")
        return None
    except Exception as e:
        print(f"  [WARN] Title {title_num:02d} — {e}, skipping.")
        return None


def _extract_xml_files(zip_path: Path) -> list[bytes]:
    """Extract all .xml files from a ZIP as bytes."""
    xml_files = []
    with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
        for name in zf.namelist():
            if name.endswith(".xml") and not name.startswith("__"):
                xml_files.append(zf.read(name))
    return xml_files


def _clean_text(raw: str) -> str:
    """Normalise whitespace and remove artefacts from extracted XML text."""
    text = re.sub(r"\s+", " ", raw)
    text = re.sub(r"\[\d+\]", "", text)       # footnote markers like [1]
    text = re.sub(r"\(Pub\. L\..*?\)", "", text)  # public law references in parens
    return text.strip()


def _get_all_text(element) -> str:
    """Concatenate all text nodes inside an element, regardless of namespace."""
    return " ".join(element.itertext())


def _parse_sections(xml_bytes: bytes, title_num: int) -> list[dict]:
    """
    Parse a US Code XML file (USLM schema) and return a list of raw section dicts.

    Each dict has: title_num, title_name, section_num, section_name, text, citation
    """
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError as e:
        print(f"  [WARN] XML parse error in title {title_num:02d}: {e}")
        return []

    title_name = TITLE_NAMES.get(title_num, f"Title {title_num}")
    sections = []

    # USLM uses namespaced elements — {*}section matches any namespace
    for section_el in root.iter("{*}section"):
        # Section number
        num_el = next(section_el.iter("{*}num"), None)
        section_num = num_el.get("value", "").strip() if num_el is not None else ""
        if not section_num:
            # Try text content of num element
            section_num = (num_el.text or "").strip() if num_el is not None else ""

        # Section heading
        heading_el = next(section_el.iter("{*}heading"), None)
        section_name = _clean_text(_get_all_text(heading_el)) if heading_el is not None else ""

        # Full text of section (all nested text)
        full_text = _clean_text(_get_all_text(section_el))

        if not full_text or len(full_text) < 30:
            continue

        # Build formal citation: e.g. "18 U.S.C. § 1341"
        citation = f"{title_num} U.S.C. § {section_num}" if section_num else f"Title {title_num}"

        sections.append({
            "title_num":    title_num,
            "title_name":   title_name,
            "section_num":  section_num,
            "section_name": section_name,
            "text":         full_text,
            "citation":     citation,
        })

    return sections


def _sections_to_chunks(sections: list[dict]) -> list[dict]:
    """Convert parsed sections to chunks using the shared chunker."""
    all_chunks = []
    for sec in sections:
        metadata = {
            "source":       "us_code",
            "jurisdiction": "us",
            "title_num":    sec["title_num"],
            "title_name":   sec["title_name"],
            "section_num":  sec["section_num"],
            "section_name": sec["section_name"],
            "citation":     sec["citation"],
        }
        chunks = chunk_text(sec["text"], metadata)
        all_chunks.extend(chunks)
    return all_chunks


# ---------------------------------------------------------------------------
# Main ingestion
# ---------------------------------------------------------------------------

def ingest(titles: list[int] | None = None) -> int:
    """
    Ingest US Code titles into data/processed/us_statutes.jsonl.

    Args:
        titles: list of title numbers to ingest (default: all 1–54)

    Returns:
        total number of chunks written
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if titles is None:
        titles = list(range(1, 55))

    total_chunks = 0

    with jsonlines.open(OUT_FILE, mode="w") as writer:
        for title_num in tqdm(titles, desc="US Code titles"):

            if TITLE_NAMES.get(title_num) is None:
                tqdm.write(f"  Title {title_num:02d} — reserved, skipping.")
                continue

            tqdm.write(f"  Title {title_num:02d} — {TITLE_NAMES[title_num]}")

            zip_path = _download_zip(title_num, RAW_DIR)
            if zip_path is None:
                continue

            xml_files = _extract_xml_files(zip_path)
            if not xml_files:
                tqdm.write(f"  [WARN] No XML in ZIP for title {title_num:02d}")
                continue

            for xml_bytes in xml_files:
                sections = _parse_sections(xml_bytes, title_num)
                chunks = _sections_to_chunks(sections)
                for chunk in chunks:
                    writer.write(chunk)
                total_chunks += len(chunks)

            tqdm.write(f"    → {total_chunks:,} total chunks so far")

    print(f"\nDone. {total_chunks:,} chunks written to {OUT_FILE}")
    return total_chunks


if __name__ == "__main__":
    ingest()
