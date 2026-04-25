"""
Phase 2 — CourtListener SCOTUS Ingestion

Fetches all SCOTUS opinions (majority + dissenting + concurring) from the
CourtListener free API, cleans HTML, chunks with overlap, and writes to
data/processed/us_case_law.jsonl

Rate limits:
    Anonymous:         100 requests/day  (too slow — DO NOT use)
    Free token:      5,000 requests/day  (~6 days for 28k cases)
    Premium token: 100,000 requests/day  (~4 hours for 28k cases)

    Get a free token at: https://www.courtlistener.com/sign-in/
    Add to .env: COURTLISTENER_TOKEN=your_token_here

Resume support:
    Progress is saved after every page to data/raw/court_listener/progress.json
    Re-run the script and it continues exactly where it left off.

Run:
    python -m src.ingestion.court_listener

Output:
    data/raw/court_listener/     — cached raw API pages (JSON)
    data/processed/us_case_law.jsonl — ~500k chunks, ~150MB
"""

import json
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

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RAW_DIR     = Path("data/raw/court_listener")
OUT_FILE    = Path("data/processed/us_case_law.jsonl")
PROGRESS    = RAW_DIR / "progress.json"

BASE_URL    = "https://www.courtlistener.com/api/rest/v3"
TOKEN       = os.getenv("COURTLISTENER_TOKEN", "")
PAGE_SIZE   = 20          # CourtListener max is 20 for v3
SLEEP_SEC   = 0.5         # between every request — adjust if rate limited

# Opinion type codes → human-readable labels
OPINION_TYPE_MAP = {
    "010": "majority",
    "015": "unanimous",
    "020": "concurrence",
    "025": "concurrence_in_part",
    "030": "dissent",
    "040": "dissent_in_part",
    "050": "plurality",
    "060": "per_curiam",
    "065": "lead",
    "070": "combined",
}

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _headers() -> dict:
    h = {"Accept": "application/json"}
    if TOKEN:
        h["Authorization"] = f"Token {TOKEN}"
    return h


def _get(url: str, params: dict | None = None, retries: int = 5) -> dict:
    """GET with exponential backoff on rate limit (429) or server errors (5xx)."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=_headers(), timeout=30)
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
            return resp.json()
        except requests.RequestException as e:
            wait = 2 ** attempt * 5
            tqdm.write(f"  [ERROR] {e} — waiting {wait}s (attempt {attempt+1}/{retries})")
            time.sleep(wait)
    return {}

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_opinion_text(raw: str) -> str:
    """Strip HTML tags and normalise whitespace from opinion text."""
    if not raw:
        return ""
    # Remove HTML
    soup = BeautifulSoup(raw, "html.parser")
    text = soup.get_text(separator=" ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove page markers like "Page 347 U.S. 483"
    text = re.sub(r"Page \d+ U\.S\. \d+", "", text)
    return text.strip()


def _pick_opinion_text(opinion: dict) -> str:
    """Pick the best available text field from an opinion object."""
    for field in ("html_lawbox", "html_columbia", "html", "plain_text", "html_anon_2020"):
        val = opinion.get(field, "")
        if val and len(val) > 100:
            return _clean_opinion_text(val) if field.startswith("html") else val.strip()
    return ""

# ---------------------------------------------------------------------------
# Progress tracking (for resumable runs)
# ---------------------------------------------------------------------------

def _load_progress() -> dict:
    if PROGRESS.exists():
        return json.loads(PROGRESS.read_text())
    return {"next_url": None, "clusters_done": [], "total_chunks": 0}


def _save_progress(state: dict) -> None:
    PROGRESS.write_text(json.dumps(state))

# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

def _extract_citation(cluster: dict) -> str:
    """Return the first reporter citation e.g. '347 U.S. 483'."""
    citations = cluster.get("citations", [])
    for c in citations:
        if isinstance(c, dict):
            volume   = c.get("volume", "")
            reporter = c.get("reporter", "")
            page     = c.get("page", "")
            if volume and reporter and page:
                return f"{volume} {reporter} {page}"
        elif isinstance(c, str):
            return c
    return cluster.get("case_name", "Unknown")

# ---------------------------------------------------------------------------
# Core ingestion
# ---------------------------------------------------------------------------

def _fetch_opinions_for_cluster(cluster_id: int) -> list[dict]:
    """Fetch all opinions for a given cluster ID."""
    data = _get(f"{BASE_URL}/opinions/", params={"cluster": cluster_id, "page_size": 20})
    return data.get("results", [])


def _process_cluster(cluster: dict, writer: jsonlines.Writer) -> int:
    """Process one SCOTUS case cluster — fetch opinions, chunk, write. Returns chunk count."""
    cluster_id  = cluster["id"]
    case_name   = cluster.get("case_name", "Unknown Case")
    date_filed  = cluster.get("date_filed", "")
    year        = int(date_filed[:4]) if date_filed else 0
    judges      = cluster.get("judges", "")
    citation    = _extract_citation(cluster)

    opinions = _fetch_opinions_for_cluster(cluster_id)
    if not opinions:
        return 0

    chunk_count = 0
    for opinion in opinions:
        opinion_type_code = str(opinion.get("type", "010"))
        opinion_type      = OPINION_TYPE_MAP.get(opinion_type_code, "majority")

        text = _pick_opinion_text(opinion)
        if not text or len(text) < 100:
            continue

        metadata = {
            "source":        "court_listener",
            "jurisdiction":  "us",
            "collection":    "us_case_law",
            "case_name":     case_name,
            "citation":      citation,
            "court":         "scotus",
            "date_filed":    date_filed,
            "year":          year,
            "judges":        judges,
            "opinion_type":  opinion_type,
        }

        chunks = chunk_text(text, metadata)
        for chunk in chunks:
            writer.write(chunk)
        chunk_count += len(chunks)

    return chunk_count


def ingest() -> int:
    """
    Ingest all SCOTUS opinions into data/processed/us_case_law.jsonl.
    Resumable — re-run to continue after interruption.

    Returns:
        total number of chunks written this run
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not TOKEN:
        print(
            "\n[WARN] COURTLISTENER_TOKEN not set in .env\n"
            "       Anonymous access = 100 requests/day — very slow.\n"
            "       Get a free token at https://www.courtlistener.com/sign-in/\n"
            "       Add to .env: COURTLISTENER_TOKEN=your_token_here\n"
        )

    state = _load_progress()

    # Starting URL — either resume from saved next_url or start fresh
    if state["next_url"]:
        next_url = state["next_url"]
        print(f"Resuming from: {next_url}")
    else:
        next_url = f"{BASE_URL}/clusters/"
        state["clusters_done"] = []
        state["total_chunks"]  = 0

    clusters_done: set = set(state.get("clusters_done", []))
    total_chunks = state.get("total_chunks", 0)

    # Open JSONL in append mode so resume doesn't overwrite existing chunks
    write_mode = "a" if OUT_FILE.exists() and state["next_url"] else "w"

    with jsonlines.open(OUT_FILE, mode=write_mode) as writer:
        params = {
            "court":      "scotus",
            "order_by":   "date_filed",
            "page_size":  PAGE_SIZE,
        }

        pbar = tqdm(desc="SCOTUS clusters", unit="case")

        while next_url:
            # Fetch a page of clusters
            if next_url == f"{BASE_URL}/clusters/":
                data = _get(next_url, params=params)
            else:
                data = _get(next_url)  # next_url already has params encoded

            if not data:
                tqdm.write("  [WARN] Empty response — stopping.")
                break

            clusters  = data.get("results", [])
            next_url  = data.get("next")    # None on last page

            for cluster in clusters:
                cluster_id = cluster["id"]
                if cluster_id in clusters_done:
                    pbar.update(1)
                    continue

                chunks_added = _process_cluster(cluster, writer)
                total_chunks += chunks_added
                clusters_done.add(cluster_id)
                pbar.update(1)
                pbar.set_postfix({"chunks": f"{total_chunks:,}"})

            # Save progress after every page
            state["next_url"]      = next_url
            state["clusters_done"] = list(clusters_done)
            state["total_chunks"]  = total_chunks
            _save_progress(state)

        pbar.close()

    print(f"\nDone. {total_chunks:,} chunks written to {OUT_FILE}")
    print(f"Covered {len(clusters_done):,} SCOTUS cases.")
    return total_chunks


if __name__ == "__main__":
    ingest()
