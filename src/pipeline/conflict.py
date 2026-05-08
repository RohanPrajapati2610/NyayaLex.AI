"""
Legal conflict detector.

Detects when two or more retrieved source chunks contradict each other
on the question being asked.

Two-layer approach:
  Layer 1 — NLI pairwise contradiction check (fast, local CPU)
             Uses same NLI model as faithfulness checker.
             If any chunk pair scores high on CONTRADICTION → flag.

  Layer 2 — Groq explanation (only triggered if Layer 1 finds a conflict)
             Generates a human-readable explanation of what specifically
             contradicts what, shown to the user as a warning banner.

This avoids calling Groq on every query — NLI does the cheap detection,
Groq only explains when a real conflict is found.
"""

import json

from src.llm.groq_client import chat
from src.llm.prompts import conflict_detection_prompt
from src.pipeline.faithfulness import _get_nli

CONTRADICTION_THRESHOLD = 0.65   # NLI contradiction score above this = conflict


# ---------------------------------------------------------------------------
# Layer 1 — NLI pairwise check
# ---------------------------------------------------------------------------

def _nli_contradiction_score(text_a: str, text_b: str) -> float:
    """
    Returns the NLI contradiction score between two texts.
    Score range: 0.0 (no contradiction) → 1.0 (strong contradiction).
    """
    nli = _get_nli()
    a_trunc = text_a[:800]
    b_trunc = text_b[:800]

    results = nli(f"{a_trunc} [SEP] {b_trunc}")
    scores  = {r["label"].lower(): r["score"] for r in results[0]}
    return scores.get("contradiction", 0.0)


def _find_conflicting_pair(chunks: list[dict]) -> tuple[dict, dict, float] | None:
    """
    Check all pairs of chunks for NLI contradiction.
    Returns the first conflicting pair found, or None.
    """
    texts = [(c, c.get("text", "")) for c in chunks if c.get("text")]

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            chunk_a, text_a = texts[i]
            chunk_b, text_b = texts[j]
            score = _nli_contradiction_score(text_a, text_b)
            if score >= CONTRADICTION_THRESHOLD:
                return chunk_a, chunk_b, score

    return None


# ---------------------------------------------------------------------------
# Layer 2 — Groq explanation
# ---------------------------------------------------------------------------

def _groq_conflict_explanation(question: str, chunks: list[dict]) -> str | None:
    """
    Ask Groq to explain the conflict in plain language.
    Returns explanation string or None if Groq says no conflict.
    """
    response = chat(
        messages=conflict_detection_prompt(question=question, chunks=chunks),
        temperature=0.0,
        max_tokens=200,
    )
    try:
        parsed = json.loads(response)
        if parsed.get("conflict"):
            return parsed.get("explanation", "Conflicting legal sources detected.")
        return None
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_conflict(question: str, chunks: list[dict]) -> dict:
    """
    Full two-layer conflict detection on a set of retrieved chunks.

    Args:
        question: the user's legal question
        chunks:   list of retrieved chunk dicts (with "text" and "metadata")

    Returns:
        {
          "conflict":     bool,
          "explanation":  str | None,   # shown to user as warning banner
          "chunk_a":      dict | None,  # first conflicting chunk
          "chunk_b":      dict | None,  # second conflicting chunk
          "nli_score":    float         # contradiction score 0–1
        }
    """
    if len(chunks) < 2:
        return {"conflict": False, "explanation": None,
                "chunk_a": None, "chunk_b": None, "nli_score": 0.0}

    # Layer 1 — fast NLI check
    conflict_pair = _find_conflicting_pair(chunks)

    if not conflict_pair:
        return {"conflict": False, "explanation": None,
                "chunk_a": None, "chunk_b": None, "nli_score": 0.0}

    chunk_a, chunk_b, nli_score = conflict_pair

    # Layer 2 — Groq explanation (only runs when conflict detected)
    explanation = _groq_conflict_explanation(question, [chunk_a, chunk_b])

    if not explanation:
        # NLI detected contradiction but Groq didn't confirm —
        # still flag it but with a generic message
        meta_a = chunk_a.get("metadata", {})
        meta_b = chunk_b.get("metadata", {})
        cite_a = meta_a.get("citation", "Source A")
        cite_b = meta_b.get("citation", "Source B")
        explanation = (
            f"Potential conflict detected between {cite_a} and {cite_b}. "
            "These sources may state different rules on this question. "
            "Consult a qualified lawyer to determine which applies."
        )

    return {
        "conflict":     True,
        "explanation":  explanation,
        "chunk_a":      chunk_a,
        "chunk_b":      chunk_b,
        "nli_score":    round(nli_score, 3),
    }


def format_conflict_warning(conflict_result: dict) -> str | None:
    """
    Format conflict result into a user-facing warning string.
    Returns None if no conflict.
    """
    if not conflict_result.get("conflict"):
        return None

    meta_a = conflict_result.get("chunk_a", {}).get("metadata", {})
    meta_b = conflict_result.get("chunk_b", {}).get("metadata", {})
    cite_a = meta_a.get("citation", "Source A")
    cite_b = meta_b.get("citation", "Source B")

    return (
        f"⚠️ Conflict detected between {cite_a} and {cite_b}. "
        f"{conflict_result['explanation']}"
    )
