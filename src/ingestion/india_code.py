"""
india_code.py — Phase 4 ingestion for NyayaLex.AI

Downloads the text of major Indian central acts, extracts individual sections,
chunks each section, and writes results to data/processed/india_statutes.jsonl.

Strategy (hybrid):
  1. Attempt to fetch act content via the India Code REST API
     (https://www.indiacode.nic.in/rest/acts).
  2. Fall back to Indian Kanoon act pages
     (https://indiankanoon.org/doc/<doc_id>/) when India Code is unavailable.

Raw act text is cached under data/raw/india_code/ so interrupted runs resume
without re-downloading already saved acts.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

import jsonlines
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm

from src.ingestion.chunker import chunk_text

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw" / "india_code"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "india_statutes.jsonl"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Polite crawl delay (seconds)
REQUEST_DELAY = 1.0
MAX_RETRIES = 4
BACKOFF_BASE = 2  # seconds; doubles on each retry

INDIA_CODE_API = "https://www.indiacode.nic.in/rest/acts"
INDIAN_KANOON_SEARCH = "https://api.indiankanoon.org/search/"
INDIAN_KANOON_DOC = "https://indiankanoon.org/doc/{doc_id}/"

# ---------------------------------------------------------------------------
# Act catalogue
# Each entry: act_name, act_short, year, india_code_sid (None = unknown),
# kanoon_doc_id (None = will be searched at runtime)
# ---------------------------------------------------------------------------

ACTS = [
    {
        "act_name": "Bharatiya Nyaya Sanhita 2023",
        "act_short": "BNS",
        "year": 2023,
        "kanoon_query": "Bharatiya Nyaya Sanhita 2023",
        "kanoon_doc_id": 117821042,
    },
    {
        "act_name": "Bharatiya Nagarik Suraksha Sanhita 2023",
        "act_short": "BNSS",
        "year": 2023,
        "kanoon_query": "Bharatiya Nagarik Suraksha Sanhita 2023",
        "kanoon_doc_id": 117821043,
    },
    {
        "act_name": "Bharatiya Sakshya Adhiniyam 2023",
        "act_short": "BSA",
        "year": 2023,
        "kanoon_query": "Bharatiya Sakshya Adhiniyam 2023",
        "kanoon_doc_id": 117821044,
    },
    {
        "act_name": "Indian Contract Act 1872",
        "act_short": "ICA",
        "year": 1872,
        "kanoon_query": "Indian Contract Act 1872",
        "kanoon_doc_id": 1878082,
    },
    {
        "act_name": "Companies Act 2013",
        "act_short": "CA2013",
        "year": 2013,
        "kanoon_query": "Companies Act 2013",
        "kanoon_doc_id": 63740743,
    },
    {
        "act_name": "Income Tax Act 1961",
        "act_short": "ITA",
        "year": 1961,
        "kanoon_query": "Income Tax Act 1961",
        "kanoon_doc_id": 1099038,
    },
    {
        "act_name": "Consumer Protection Act 2019",
        "act_short": "CPA2019",
        "year": 2019,
        "kanoon_query": "Consumer Protection Act 2019",
        "kanoon_doc_id": 120768045,
    },
    {
        "act_name": "Right to Information Act 2005",
        "act_short": "RTI",
        "year": 2005,
        "kanoon_query": "Right to Information Act 2005",
        "kanoon_doc_id": 1307922,
    },
    {
        "act_name": "Protection of Children from Sexual Offences Act 2012",
        "act_short": "POCSO",
        "year": 2012,
        "kanoon_query": "Protection of Children from Sexual Offences Act 2012",
        "kanoon_doc_id": 41423355,
    },
    {
        "act_name": "Narcotic Drugs and Psychotropic Substances Act 1985",
        "act_short": "NDPS",
        "year": 1985,
        "kanoon_query": "Narcotic Drugs and Psychotropic Substances Act 1985",
        "kanoon_doc_id": 1046093,
    },
    {
        "act_name": "Prevention of Corruption Act 1988",
        "act_short": "PCA",
        "year": 1988,
        "kanoon_query": "Prevention of Corruption Act 1988",
        "kanoon_doc_id": 1125049,
    },
    {
        "act_name": "Arbitration and Conciliation Act 1996",
        "act_short": "ACA",
        "year": 1996,
        "kanoon_query": "Arbitration and Conciliation Act 1996",
        "kanoon_doc_id": 1052228,
    },
    {
        "act_name": "Transfer of Property Act 1882",
        "act_short": "TPA",
        "year": 1882,
        "kanoon_query": "Transfer of Property Act 1882",
        "kanoon_doc_id": 1222182,
    },
    {
        "act_name": "Code of Civil Procedure 1908",
        "act_short": "CPC",
        "year": 1908,
        "kanoon_query": "Code of Civil Procedure 1908",
        "kanoon_doc_id": 1714720,
    },
    {
        "act_name": "Specific Relief Act 1963",
        "act_short": "SRA",
        "year": 1963,
        "kanoon_query": "Specific Relief Act 1963",
        "kanoon_doc_id": 1240182,
    },
]

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_SESSION.headers.update(
    {
        "User-Agent": (
            "NyayaLex-Ingestion/1.0 (legal research; "
            "contact: research@nyayalex.ai)"
        ),
        "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8",
    }
)


def _get(url: str, params: dict | None = None, timeout: int = 30) -> requests.Response | None:
    """
    GET a URL with exponential backoff on HTTP 4xx/5xx and connection errors.
    Returns a Response on success, None if all retries fail.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = _SESSION.get(url, params=params, timeout=timeout)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (403, 404):
                log.warning("HTTP %s for %s — skipping", resp.status_code, url)
                return None
            wait = BACKOFF_BASE ** attempt
            log.warning(
                "HTTP %s for %s — retry %d/%d in %ds",
                resp.status_code,
                url,
                attempt,
                MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
        except requests.RequestException as exc:
            wait = BACKOFF_BASE ** attempt
            log.warning(
                "Request error for %s: %s — retry %d/%d in %ds",
                url,
                exc,
                attempt,
                MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
    log.error("All retries exhausted for %s", url)
    return None


# ---------------------------------------------------------------------------
# India Code REST API
# ---------------------------------------------------------------------------


def _fetch_via_india_code(act_info: dict) -> str | None:
    """
    Attempt to pull act text from the India Code REST API.
    Returns plain text on success, None on failure.
    """
    try:
        resp = _get(INDIA_CODE_API, timeout=20)
        if resp is None:
            return None
        acts_list = resp.json()
        if not isinstance(acts_list, list):
            return None

        # Match by title similarity
        target = act_info["act_name"].lower()
        matched = None
        for item in acts_list:
            title = (item.get("title") or item.get("actTitle") or "").lower()
            if act_info["act_short"].lower() in title or target[:20] in title:
                matched = item
                break

        if matched is None:
            return None

        # Try fetching act detail URL from matched item
        act_url = matched.get("url") or matched.get("actUrl")
        if not act_url:
            return None

        if not act_url.startswith("http"):
            act_url = "https://www.indiacode.nic.in" + act_url

        time.sleep(REQUEST_DELAY)
        detail_resp = _get(act_url)
        if detail_resp is None:
            return None

        soup = BeautifulSoup(detail_resp.text, "lxml")
        body = soup.find("div", class_=re.compile(r"act-content|content|body", re.I))
        if body is None:
            body = soup.find("body")
        return body.get_text(separator="\n") if body else None

    except Exception as exc:  # noqa: BLE001
        log.debug("India Code API error for %s: %s", act_info["act_name"], exc)
        return None


# ---------------------------------------------------------------------------
# Indian Kanoon fallback
# ---------------------------------------------------------------------------


def _search_kanoon_doc_id(query: str) -> int | None:
    """
    Search Indian Kanoon for an act and return the doc_id of the first result.
    Uses the public search API.
    """
    time.sleep(REQUEST_DELAY)
    resp = _get(
        INDIAN_KANOON_SEARCH,
        params={"formInput": query, "pagenum": 0},
        timeout=30,
    )
    if resp is None:
        return None
    try:
        data = resp.json()
        docs = data.get("docs") or []
        if docs:
            tid = docs[0].get("tid")
            return int(tid) if tid else None
    except Exception:  # noqa: BLE001
        pass
    return None


def _fetch_kanoon_doc(doc_id: int) -> str | None:
    """
    Fetch act text from an Indian Kanoon document page.
    Returns extracted plain text, or None on failure.
    """
    url = INDIAN_KANOON_DOC.format(doc_id=doc_id)
    time.sleep(REQUEST_DELAY)
    resp = _get(url)
    if resp is None:
        return None

    soup = BeautifulSoup(resp.text, "lxml")

    # Remove script / style noise
    for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
        tag.decompose()

    # Indian Kanoon wraps act text in <div class="doc"> or <div id="judgment">
    content = (
        soup.find("div", id="judgment")
        or soup.find("div", class_="doc")
        or soup.find("div", class_=re.compile(r"judgments?|document|content", re.I))
        or soup.find("body")
    )

    if content is None:
        return None

    return content.get_text(separator="\n")


def _fetch_act_text(act_info: dict) -> str | None:
    """
    Return raw act text, trying India Code first and Indian Kanoon second.
    """
    # 1. India Code REST API
    log.info("Trying India Code API for: %s", act_info["act_name"])
    text = _fetch_via_india_code(act_info)
    if text and len(text.strip()) > 500:
        log.info("India Code API succeeded for: %s", act_info["act_name"])
        return text

    # 2. Indian Kanoon — known doc_id
    doc_id = act_info.get("kanoon_doc_id")
    if doc_id:
        log.info("Fetching from Indian Kanoon (doc %d): %s", doc_id, act_info["act_name"])
        text = _fetch_kanoon_doc(doc_id)
        if text and len(text.strip()) > 500:
            return text

    # 3. Indian Kanoon — search fallback
    log.info("Searching Indian Kanoon for: %s", act_info["kanoon_query"])
    found_id = _search_kanoon_doc_id(act_info["kanoon_query"])
    if found_id and found_id != doc_id:
        log.info("Search found doc_id %d — fetching", found_id)
        text = _fetch_kanoon_doc(found_id)
        if text and len(text.strip()) > 500:
            return text

    log.warning("Could not fetch act text for: %s", act_info["act_name"])
    return None


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------

# Matches patterns like:
#   "Section 1." / "1." / "1 ." / "Section 1 -" etc.
_SECTION_RE = re.compile(
    r"(?:^|\n)\s*(?:Section\s+)?(\d+[A-Z]?)\s*[.)\-\u2013\u2014]\s*([^\n]{0,120})\n",
    re.MULTILINE,
)


def _parse_sections(raw_text: str) -> list[dict]:
    """
    Extract (section_num, section_heading, section_text) triples from raw act text.

    Heuristic: locate section headings via _SECTION_RE, then treat everything
    between consecutive headings as that section's body text.

    Returns a list of dicts with keys: section_num, section_heading, text.
    """
    raw_text = re.sub(r"\r\n?", "\n", raw_text)
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)

    matches = list(_SECTION_RE.finditer(raw_text))
    if not matches:
        # No structured sections found — treat entire text as one block
        return [{"section_num": "1", "section_heading": "Full Text", "text": raw_text.strip()}]

    sections = []
    for i, match in enumerate(matches):
        sec_num = match.group(1).strip()
        heading_raw = match.group(2).strip()
        # Clean heading: strip trailing punctuation noise
        heading = re.sub(r"[.\-\u2013\u2014]+$", "", heading_raw).strip()

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()

        # Skip near-empty sections
        if len(body) < 20:
            continue

        sections.append(
            {
                "section_num": sec_num,
                "section_heading": heading or f"Section {sec_num}",
                "text": body,
            }
        )

    return sections


# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------


def _safe_filename(name: str) -> str:
    return re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")


def ingest() -> int:
    """
    Download, parse, chunk, and persist all major Indian central acts.

    Returns the total number of chunks written to data/processed/india_statutes.jsonl.
    """
    total_chunks = 0

    with jsonlines.open(OUTPUT_FILE, mode="w") as writer:
        for act_info in tqdm(ACTS, desc="Acts", unit="act"):
            act_name = act_info["act_name"]
            safe_name = _safe_filename(act_name)
            raw_path = RAW_DIR / f"{safe_name}.txt"

            # --- Resume: load from cache if already downloaded ---
            if raw_path.exists():
                log.info("Cache hit — loading from %s", raw_path)
                raw_text = raw_path.read_text(encoding="utf-8")
            else:
                raw_text = _fetch_act_text(act_info)
                if not raw_text:
                    log.error("Skipping %s — no text obtained", act_name)
                    continue
                raw_path.write_text(raw_text, encoding="utf-8")
                log.info("Saved raw text for %s (%d chars)", act_name, len(raw_text))
                time.sleep(REQUEST_DELAY)

            # --- Parse sections ---
            sections = _parse_sections(raw_text)
            log.info(
                "%s — %d sections found", act_name, len(sections)
            )

            # --- Chunk each section ---
            act_chunks = 0
            for section in tqdm(
                sections,
                desc=act_info["act_short"],
                leave=False,
                unit="sec",
            ):
                citation = (
                    f"{act_info['act_short']} § {section['section_num']}"
                )
                metadata = {
                    "source": "india_code",
                    "jurisdiction": "india",
                    "collection": "india_statutes",
                    "act_name": act_name,
                    "act_short": act_info["act_short"],
                    "year": act_info["year"],
                    "section_num": section["section_num"],
                    "section_heading": section["section_heading"],
                    "citation": citation,
                }

                chunks = chunk_text(section["text"], metadata)
                for chunk in chunks:
                    writer.write(chunk)
                    act_chunks += 1

            log.info(
                "%s — %d chunks written", act_name, act_chunks
            )
            total_chunks += act_chunks

    log.info("Ingestion complete. Total chunks: %d", total_chunks)
    log.info("Output: %s", OUTPUT_FILE)
    return total_chunks


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ingest()
