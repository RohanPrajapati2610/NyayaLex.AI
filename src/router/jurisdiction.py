"""
Jurisdiction router — maps a legal question to the correct ChromaDB
and BM25 collections to search.

Three jurisdictions:
  US    → us_statutes, us_case_law, us_regulations
  INDIA → india_statutes, india_constitution, india_case_law
  BOTH  → all 6 collections

The router uses Groq zero-shot classification first, then falls back
to keyword heuristics if Groq is unavailable.
"""

import re

from src.llm.groq_client import chat
from src.llm.prompts import jurisdiction_prompt

# ---------------------------------------------------------------------------
# Collection mapping
# ---------------------------------------------------------------------------

COLLECTIONS_US = [
    "us_statutes",
    "us_case_law",
    "us_regulations",
]

COLLECTIONS_INDIA = [
    "india_statutes",
    "india_constitution",
    "india_case_law",
]

COLLECTIONS_ALL = COLLECTIONS_US + COLLECTIONS_INDIA

JURISDICTION_MAP = {
    "US":    COLLECTIONS_US,
    "INDIA": COLLECTIONS_INDIA,
    "BOTH":  COLLECTIONS_ALL,
}

# ---------------------------------------------------------------------------
# Keyword heuristics (fallback when Groq unavailable)
# ---------------------------------------------------------------------------

_INDIA_KEYWORDS = re.compile(
    r"\b(india|indian|bns|bnss|bsa|ipc|crpc|constitution of india|"
    r"supreme court of india|high court|article \d+|schedule [ivx]+|"
    r"bharat|lok sabha|rajya sabha|rbi|sebi|rti|pocso|ndps|"
    r"preamble|fundamental rights?|directive principles?)\b",
    re.IGNORECASE,
)

_US_KEYWORDS = re.compile(
    r"\b(united states|u\.s\.|us code|usc|cfr|scotus|federal|congress|"
    r"amendment|bill of rights|constitution of the united states|"
    r"title \d+|section \d+|irs|fda|osha|epa|nlra|flsa|erisa|"
    r"first amendment|second amendment|fourteenth amendment)\b",
    re.IGNORECASE,
)


def _heuristic_jurisdiction(question: str) -> str:
    has_india = bool(_INDIA_KEYWORDS.search(question))
    has_us    = bool(_US_KEYWORDS.search(question))

    if has_india and has_us:
        return "BOTH"
    if has_india:
        return "INDIA"
    if has_us:
        return "US"
    return "BOTH"  # default — search everything when ambiguous


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_jurisdiction(question: str) -> str:
    """
    Detect jurisdiction of a legal question.

    Returns one of: "US", "INDIA", "BOTH"
    Tries Groq first; falls back to keyword heuristics on error.
    """
    try:
        response = chat(
            messages=jurisdiction_prompt(question),
            temperature=0.0,
            max_tokens=10,
        )
        result = response.strip().upper()
        if result in ("US", "INDIA", "BOTH"):
            return result
        # Groq returned something unexpected — use heuristics
        return _heuristic_jurisdiction(question)
    except Exception:
        return _heuristic_jurisdiction(question)


def get_collections(question: str) -> tuple[str, list[str]]:
    """
    Returns (jurisdiction, collection_names) for a question.

    Example:
        jurisdiction, collections = get_collections("Can a tenant be evicted?")
        # → ("US", ["us_statutes", "us_case_law", "us_regulations"])
    """
    jurisdiction = detect_jurisdiction(question)
    return jurisdiction, JURISDICTION_MAP[jurisdiction]


def collections_for_jurisdiction(jurisdiction: str) -> list[str]:
    """Return collection names for a known jurisdiction string."""
    return JURISDICTION_MAP.get(jurisdiction.upper(), COLLECTIONS_ALL)
