"""
india_kanoon.py — Supreme Court of India judgment ingestion for NyayaLex.AI

What it does:
    Fetches Supreme Court of India judgments from the Indian Kanoon API,
    chunks each judgment's text, and writes the results to
    data/processed/india_case_law.jsonl.

    Up to 5,000 judgments are fetched (100 pages × 10 results per page,
    then repeated across multiple search passes with date-range refinements
    until the target is reached or the API is exhausted).

API used:
    Indian Kanoon API — https://api.indiankanoon.org/
    Requires a free API token set as INDIAN_KANOON_TOKEN in .env

    Search endpoint  : POST https://api.indiankanoon.org/search/
        Body params  : formInput, pagenum
    Document endpoint: POST https://api.indiankanoon.org/doc/{doc_id}/

How to run:
    1. Add  INDIAN_KANOON_TOKEN=<your_token>  to your .env file
       (Get a free token at https://api.indiankanoon.org/)
    2. python -m src.ingestion.india_kanoon

Output files:
    data/raw/indian_kanoon/progress.json           — resume state
    data/raw/indian_kanoon/{doc_id}.json           — raw API response per judgment
    data/processed/india_case_law.jsonl            — one JSON object per chunk
"""

import json
import os
import re
import time
from pathlib import Path

import jsonlines
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from src.ingestion.chunker import chunk_text

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_TOKEN = os.getenv("INDIAN_KANOON_TOKEN", "")

BASE_URL = "https://api.indiankanoon.org"
SEARCH_URL = f"{BASE_URL}/search/"
DOC_URL_TPL = f"{BASE_URL}/doc/{{doc_id}}/"

RAW_DIR = Path("data/raw/indian_kanoon")
PROCESSED_DIR = Path("data/processed")
PROGRESS_FILE = RAW_DIR / "progress.json"
OUT_FILE = PROCESSED_DIR / "india_case_law.jsonl"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DOCS = 5_000
RESULTS_PER_PAGE = 10          # Indian Kanoon returns 10 results per page
SLEEP_BETWEEN = 0.5            # seconds between every HTTP request
MAX_RETRIES = 5
BACKOFF_BASE = 2.0             # seconds, exponential backoff

# Search passes — each is a (formInput, label) pair.
# Multiple passes cover different time windows so we collect more judgments.
SEARCH_PASSES = [
    (
        "doctypes:judgments court:\"Supreme Court of India\"",
        "SC judgments (all)",
    ),
    (
        "doctypes:judgments court:\"Supreme Court of India\" fromdate:2010-01-01 todate:2023-12-31",
        "SC judgments 2010-2023",
    ),
    (
        "doctypes:judgments court:\"Supreme Court of India\" fromdate:2000-01-01 todate:2009-12-31",
        "SC judgments 2000-2009",
    ),
    (
        "doctypes:judgments court:\"Supreme Court of India\" fromdate:1990-01-01 todate:1999-12-31",
        "SC judgments 1990-1999",
    ),
    (
        "doctypes:judgments court:\"Supreme Court of India\" fromdate:1950-01-01 todate:1989-12-31",
        "SC judgments 1950-1989",
    ),
]


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {
        "Authorization": f"Token {API_TOKEN}",
        "Content-Type": "application/x-www-form-urlencoded",
    }


def _post(url: str, data: dict) -> dict | None:
    """POST with exponential backoff on 429 / 5xx. Returns parsed JSON or None."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, data=data, headers=_headers(), timeout=60)
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError:
                    return None
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = BACKOFF_BASE ** attempt
                time.sleep(wait)
                continue
            # 401 / 403 — bad token, abort immediately
            if resp.status_code in (401, 403):
                raise RuntimeError(
                    f"Indian Kanoon API returned {resp.status_code}. "
                    "Check INDIAN_KANOON_TOKEN in your .env file."
                )
            return None
        except requests.RequestException:
            wait = BACKOFF_BASE ** attempt
            time.sleep(wait)
    return None


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

def _load_progress() -> dict:
    """Load resume state from disk. Returns empty state if file missing."""
    if PROGRESS_FILE.exists():
        try:
            with PROGRESS_FILE.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "fetched_ids": [],          # list of doc_ids already processed
        "total_chunks": 0,
    }


def _save_progress(state: dict) -> None:
    with PROGRESS_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Text / metadata extraction
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_YEAR_RE = re.compile(r"\b(\d{4})\b")


def _extract_date(doc: dict) -> str:
    """Try to pull an ISO date from the document metadata."""
    for field in ("publishdate", "date", "judgment_date"):
        val = doc.get(field, "")
        if val:
            m = _DATE_RE.search(str(val))
            if m:
                return m.group(1)
    return ""


def _extract_year(date_str: str, doc: dict) -> int | None:
    if date_str:
        m = _YEAR_RE.search(date_str)
        if m:
            return int(m.group(1))
    for field in ("year", "pubyear"):
        val = doc.get(field)
        if val:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


def _extract_citation(doc: dict) -> str:
    """Return the best available citation string."""
    for field in ("citation", "equiv", "cites"):
        val = doc.get(field, "")
        if isinstance(val, list):
            val = "; ".join(val)
        if val:
            return str(val).strip()
    return ""


def _extract_text(doc: dict) -> str:
    """Extract the main judgment text from the API response."""
    # The API returns a 'doc' key with HTML or plain text
    raw = doc.get("doc", "") or doc.get("judgment", "") or doc.get("text", "")
    if not raw:
        return ""
    # Strip HTML tags if present
    if "<" in raw:
        raw = re.sub(r"<[^>]+>", " ", raw)
    # Normalise whitespace
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def _build_metadata(doc: dict, doc_id: str) -> dict:
    date_str = _extract_date(doc)
    year = _extract_year(date_str, doc)
    return {
        "source": "indian_kanoon",
        "jurisdiction": "india",
        "collection": "india_case_law",
        "case_name": (doc.get("title") or doc.get("docsource") or "").strip(),
        "court": "Supreme Court of India",
        "date": date_str,
        "year": year,
        "citation": _extract_citation(doc),
        "doc_id": doc_id,
    }


# ---------------------------------------------------------------------------
# Core fetch routines
# ---------------------------------------------------------------------------

def _fetch_doc(doc_id: str) -> dict | None:
    """Fetch a single document from the Indian Kanoon doc endpoint."""
    raw_path = RAW_DIR / f"{doc_id}.json"
    if raw_path.exists():
        try:
            with raw_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass  # re-fetch if cached file is corrupt

    data = _post(DOC_URL_TPL.format(doc_id=doc_id), {})
    if data:
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    return data


def _search_page(form_input: str, pagenum: int) -> list[str]:
    """
    Run one search page and return a list of doc_id strings.
    Returns empty list if no results or API error.
    """
    data = _post(SEARCH_URL, {"formInput": form_input, "pagenum": pagenum})
    if not data:
        return []

    docs = data.get("docs", [])
    ids: list[str] = []
    for item in docs:
        # Each item has a 'tid' (document id)
        tid = item.get("tid") or item.get("docid") or item.get("id")
        if tid:
            ids.append(str(tid))
    return ids


# ---------------------------------------------------------------------------
# Main ingestion entry point
# ---------------------------------------------------------------------------

def ingest() -> int:
    """
    Fetch up to 5,000 Supreme Court of India judgments from Indian Kanoon,
    chunk each judgment, and write results to data/processed/india_case_law.jsonl.

    Resumes from where it left off using data/raw/indian_kanoon/progress.json.

    Returns the total number of chunks written in this run.
    """
    if not API_TOKEN:
        raise RuntimeError(
            "INDIAN_KANOON_TOKEN is not set. "
            "Add it to your .env file (get a free token at https://api.indiankanoon.org/)."
        )

    state = _load_progress()
    fetched_ids: set[str] = set(state.get("fetched_ids", []))
    total_chunks: int = state.get("total_chunks", 0)

    # Append mode so we don't overwrite chunks from a previous run
    with jsonlines.open(OUT_FILE, mode="a") as writer:

        outer_bar = tqdm(
            total=TARGET_DOCS,
            initial=len(fetched_ids),
            desc="SC India judgments",
            unit="doc",
        )

        for form_input, pass_label in SEARCH_PASSES:
            if len(fetched_ids) >= TARGET_DOCS:
                break

            tqdm.write(f"\n[Pass] {pass_label}")

            # Indian Kanoon search API supports pagenum 0..N.
            # We iterate pages until we hit the target or exhaust results.
            pagenum = 0
            consecutive_empty = 0

            while len(fetched_ids) < TARGET_DOCS:
                doc_ids = _search_page(form_input, pagenum)
                time.sleep(SLEEP_BETWEEN)

                if not doc_ids:
                    consecutive_empty += 1
                    if consecutive_empty >= 3:
                        break  # No more results for this search pass
                    pagenum += 1
                    continue

                consecutive_empty = 0

                for doc_id in doc_ids:
                    if len(fetched_ids) >= TARGET_DOCS:
                        break
                    if doc_id in fetched_ids:
                        continue

                    doc = _fetch_doc(doc_id)
                    time.sleep(SLEEP_BETWEEN)

                    if not doc:
                        continue

                    text = _extract_text(doc)
                    if not text:
                        # Mark as fetched even if empty to avoid retry loops
                        fetched_ids.add(doc_id)
                        continue

                    metadata = _build_metadata(doc, doc_id)
                    chunks = chunk_text(text, metadata)

                    for chunk in chunks:
                        writer.write(chunk)

                    total_chunks += len(chunks)
                    fetched_ids.add(doc_id)
                    outer_bar.update(1)

                    # Persist progress every 50 documents
                    if len(fetched_ids) % 50 == 0:
                        state["fetched_ids"] = list(fetched_ids)
                        state["total_chunks"] = total_chunks
                        _save_progress(state)

                pagenum += 1

        outer_bar.close()

    # Final progress save
    state["fetched_ids"] = list(fetched_ids)
    state["total_chunks"] = total_chunks
    _save_progress(state)

    print(f"\nDone. {len(fetched_ids)} documents fetched, {total_chunks} chunks written to {OUT_FILE}")
    return total_chunks


if __name__ == "__main__":
    ingest()
