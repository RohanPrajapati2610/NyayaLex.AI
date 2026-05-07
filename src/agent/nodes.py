"""
LangGraph node implementations.

Graph flow: reason → retrieve → check → generate

  REASON   — LLM reads all chunks retrieved so far, decides next search query + collections
  RETRIEVE — runs hybrid RAG (HyDE → dense + BM25 → RRF → reranker) on chosen collections
  CHECK    — LLM decides if enough info to answer, or another hop is needed
  GENERATE — LLM synthesises final answer with inline citations + conflict detection
"""

import json

from src.agent.state import LegalResearchState
from src.llm.groq_client import chat
from src.llm.prompts import (
    check_prompt,
    conflict_detection_prompt,
    generate_prompt,
    reason_prompt,
)
from src.router.jurisdiction import collections_for_jurisdiction


# ---------------------------------------------------------------------------
# REASON node
# ---------------------------------------------------------------------------

def reason_node(state: LegalResearchState) -> dict:
    """
    Reads all retrieved chunks so far and decides:
    - What to search next (next_query)
    - Which collections to search (next_collections)
    """
    response = chat(
        messages=reason_prompt(
            question=state["question"],
            hops_so_far=state["hops"],
            conversation_summary=state.get("conversation_history", ""),
        ),
        temperature=0.0,
        max_tokens=256,
    )

    try:
        parsed      = json.loads(response)
        next_query  = parsed.get("search_query", state["question"])
        collections = parsed.get("collections", collections_for_jurisdiction(state["jurisdiction"]))
        reasoning   = parsed.get("reasoning", "")
    except (json.JSONDecodeError, ValueError):
        # Fallback — search original question across all jurisdiction collections
        next_query  = state["question"]
        collections = collections_for_jurisdiction(state["jurisdiction"])
        reasoning   = "fallback to default search"

    # Include uploaded doc collection if present
    if state.get("uploaded_doc_collection") and "uploaded_doc" not in collections:
        collections = list(collections) + ["uploaded_doc"]

    return {
        "next_query":       next_query,
        "next_collections": collections,
        "reasoning_trace":  [reasoning],
    }


# ---------------------------------------------------------------------------
# RETRIEVE node
# ---------------------------------------------------------------------------

def retrieve_node(state: LegalResearchState, tools: dict) -> dict:
    """
    Runs hybrid retrieval on the collections chosen by REASON.
    Accumulates retrieved chunks into state["hops"].
    """
    query       = state["next_query"]
    collections = state.get("next_collections") or collections_for_jurisdiction(state["jurisdiction"])
    history     = state.get("conversation_history", "")

    # Use "all" tool if collections match the full jurisdiction set,
    # otherwise call per-collection tools and merge
    all_chunks: list[dict] = []
    for col in collections:
        tool_fn = tools.get(col) or tools.get("all")
        if tool_fn:
            chunks = tool_fn(query, history)
            all_chunks.extend(chunks)

    # Deduplicate by chunk id
    seen: set[str] = set()
    unique: list[dict] = []
    for chunk in all_chunks:
        cid = chunk.get("id", "")
        if cid not in seen:
            seen.add(cid)
            unique.append(chunk)

    # Tag each chunk with which hop it came from
    hop_record = {
        "query":       query,
        "collections": collections,
        "chunks":      unique[:10],  # cap at 10 per hop to stay within context
    }

    return {
        "hops":      [hop_record],   # Annotated[list, operator.add] — appends to existing
        "hop_count": state["hop_count"] + 1,
    }


# ---------------------------------------------------------------------------
# CHECK node
# ---------------------------------------------------------------------------

def check_node(state: LegalResearchState) -> dict:
    """
    Decides whether enough information has been retrieved to generate
    a full answer, or whether another retrieval hop is needed.
    """
    # Hard exit at max_hops
    if state["hop_count"] >= state["max_hops"]:
        return {"sufficient": True}

    response = chat(
        messages=check_prompt(
            question=state["question"],
            hops_so_far=state["hops"],
        ),
        temperature=0.0,
        max_tokens=64,
    )

    try:
        parsed     = json.loads(response)
        sufficient = bool(parsed.get("enough", False))
    except (json.JSONDecodeError, ValueError):
        sufficient = True  # default to generating rather than looping infinitely

    return {"sufficient": sufficient}


# ---------------------------------------------------------------------------
# GENERATE node
# ---------------------------------------------------------------------------

def generate_node(state: LegalResearchState) -> dict:
    """
    Synthesises all retrieved chunks into a final answer with inline citations.
    Also runs conflict detection across retrieved sources.
    """
    # Flatten all chunks from all hops
    all_chunks: list[dict] = []
    for hop in state["hops"]:
        all_chunks.extend(hop.get("chunks", []))

    # Deduplicate
    seen: set[str] = set()
    unique_chunks: list[dict] = []
    for chunk in all_chunks:
        cid = chunk.get("id", "")
        if cid not in seen:
            seen.add(cid)
            unique_chunks.append(chunk)

    top_chunks = unique_chunks[:5]  # final top-5 for answer generation

    # --- Conflict detection ---
    conflict_warning = None
    if len(top_chunks) >= 2:
        conflict_response = chat(
            messages=conflict_detection_prompt(
                question=state["question"],
                chunks=top_chunks,
            ),
            temperature=0.0,
            max_tokens=128,
        )
        try:
            conflict_data = json.loads(conflict_response)
            if conflict_data.get("conflict"):
                conflict_warning = conflict_data.get("explanation", "Conflicting sources detected.")
        except (json.JSONDecodeError, ValueError):
            pass

    # --- Generate final answer ---
    answer = chat(
        messages=generate_prompt(
            question=state["question"],
            retrieved_chunks=top_chunks,
            jurisdiction=state["jurisdiction"],
            conversation_summary=state.get("conversation_history", ""),
        ),
        temperature=0.0,
        max_tokens=1024,
        stream=False,
    )

    # --- Build citation list ---
    citations = []
    for chunk in top_chunks:
        meta = chunk.get("metadata", {})
        citations.append({
            "source":   meta.get("citation", meta.get("case_name", "Unknown")),
            "excerpt":  chunk.get("text", "")[:300],
            "court":    meta.get("court"),
            "date":     meta.get("date_filed") or meta.get("date"),
            "citation": meta.get("citation"),
            "score":    chunk.get("reranker_score", chunk.get("rrf_score", 0.0)),
            "faithful": True,  # NLI faithfulness check is applied in Phase 12 (pipeline)
        })

    return {
        "final_answer":    answer,
        "citations":       citations,
        "conflict_warning": conflict_warning,
    }
