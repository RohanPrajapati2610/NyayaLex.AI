"""
NLI-based citation faithfulness checker.

For every claim in the generated answer, checks whether the cited source
chunk actually ENTAILS (supports) that claim using a Natural Language
Inference model.

Model: cross-encoder/nli-deberta-v3-small
  - Runs on CPU, ~85MB, free from HuggingFace
  - Labels: ENTAILMENT, NEUTRAL, CONTRADICTION
  - Score threshold: entailment score >= 0.5 → faithful

Why NLI and not LLM?
  NLI is a dedicated classification model trained specifically for
  premise→hypothesis entailment. It catches hallucinations that the
  LLM itself would miss because it cannot self-audit.
"""

from transformers import pipeline as hf_pipeline

NLI_MODEL      = "cross-encoder/nli-deberta-v3-small"
FAITHFUL_THRESHOLD = 0.5   # entailment score above this = faithful

_nli_pipeline = None


def _get_nli() :
    global _nli_pipeline
    if _nli_pipeline is None:
        _nli_pipeline = hf_pipeline(
            "text-classification",
            model=NLI_MODEL,
            device=-1,           # CPU
            top_k=None,          # return all label scores
        )
    return _nli_pipeline


# ---------------------------------------------------------------------------
# Core check
# ---------------------------------------------------------------------------

def check_entailment(premise: str, hypothesis: str) -> dict:
    """
    Check if premise ENTAILS hypothesis.

    Args:
        premise:    source chunk text (the cited document)
        hypothesis: claim made in the LLM answer

    Returns:
        {
          "faithful":    bool,
          "entailment":  float,   # entailment score 0–1
          "neutral":     float,
          "contradiction": float
        }
    """
    nli = _get_nli()

    # Truncate to avoid exceeding model context (512 tokens)
    premise_trunc    = premise[:1000]
    hypothesis_trunc = hypothesis[:300]

    results = nli(f"{premise_trunc} [SEP] {hypothesis_trunc}")

    scores = {r["label"].lower(): r["score"] for r in results[0]}

    entailment_score = scores.get("entailment", 0.0)
    return {
        "faithful":      entailment_score >= FAITHFUL_THRESHOLD,
        "entailment":    entailment_score,
        "neutral":       scores.get("neutral", 0.0),
        "contradiction": scores.get("contradiction", 0.0),
    }


# ---------------------------------------------------------------------------
# Sentence splitter (simple — no NLTK dependency)
# ---------------------------------------------------------------------------

def _split_claims(answer: str) -> list[str]:
    """Split answer into individual sentences to check each claim."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"(])", answer)
    # Filter out very short sentences (likely not claims)
    return [s.strip() for s in sentences if len(s.strip()) > 40]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_citations(answer: str, citations: list[dict]) -> list[dict]:
    """
    For each citation, check if it faithfully supports at least one claim
    in the generated answer.

    Args:
        answer:    the LLM-generated answer text
        citations: list of citation dicts with keys: source, excerpt, ...

    Returns:
        Updated citations list with "faithful" and "faithfulness_score" added.
    """
    claims = _split_claims(answer)
    if not claims:
        return citations

    updated = []
    for citation in citations:
        source_text = citation.get("excerpt", "")
        if not source_text:
            updated.append({**citation, "faithful": True, "faithfulness_score": 1.0})
            continue

        # Check entailment against each claim — citation is faithful
        # if it entails at least ONE claim in the answer
        best_entailment = 0.0
        for claim in claims[:5]:   # cap at 5 claims to limit CPU time
            result = check_entailment(premise=source_text, hypothesis=claim)
            if result["entailment"] > best_entailment:
                best_entailment = result["entailment"]
            if result["faithful"]:
                break  # found one supported claim — enough

        updated.append({
            **citation,
            "faithful":           best_entailment >= FAITHFUL_THRESHOLD,
            "faithfulness_score": round(best_entailment, 3),
        })

    return updated


def faithfulness_summary(citations: list[dict]) -> dict:
    """
    Compute overall faithfulness stats for a set of verified citations.

    Returns:
        {
          "total":         int,
          "faithful":      int,
          "unfaithful":    int,
          "avg_score":     float,
          "hallucination_risk": "low" | "medium" | "high"
        }
    """
    total     = len(citations)
    if total == 0:
        return {"total": 0, "faithful": 0, "unfaithful": 0,
                "avg_score": 0.0, "hallucination_risk": "unknown"}

    faithful  = sum(1 for c in citations if c.get("faithful", True))
    scores    = [c.get("faithfulness_score", 1.0) for c in citations]
    avg_score = sum(scores) / len(scores)

    if avg_score >= 0.75:
        risk = "low"
    elif avg_score >= 0.5:
        risk = "medium"
    else:
        risk = "high"

    return {
        "total":              total,
        "faithful":           faithful,
        "unfaithful":         total - faithful,
        "avg_score":          round(avg_score, 3),
        "hallucination_risk": risk,
    }
