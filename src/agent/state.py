"""
LangGraph state for the multi-hop legal research agent.

Flows through every node: reason → retrieve → check → generate
"""

import operator
from typing import Annotated, TypedDict


class CitationResult(TypedDict):
    source:   str
    excerpt:  str
    court:    str | None
    date:     str | None
    citation: str | None
    score:    float
    faithful: bool


class OutcomePrediction(TypedDict):
    verdict:    str
    confidence: float
    label:      str


class LegalResearchState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    question:                str
    jurisdiction:            str           # "US" | "INDIA" | "BOTH"
    session_id:              str
    uploaded_doc_collection: str | None    # per-session ChromaDB collection if PDF uploaded

    # ── Multi-hop tracking ───────────────────────────────────────────────────
    hops:            Annotated[list[dict], operator.add]  # chunks accumulated across hops
    hop_count:       int
    max_hops:        int                   # default 4
    reasoning_trace: Annotated[list[str], operator.add]
    next_query:      str                   # refined query for next hop (set by REASON)
    next_collections: list[str]            # collections to search next (set by REASON)
    sufficient:      bool                  # CHECK node sets True to exit loop

    # ── Memory ───────────────────────────────────────────────────────────────
    conversation_history: str              # compressed summary from ConversationSummaryMemory

    # ── Output ───────────────────────────────────────────────────────────────
    final_answer:    str
    citations:       list[CitationResult]
    conflict_warning: str | None
    outcome:         OutcomePrediction | None
