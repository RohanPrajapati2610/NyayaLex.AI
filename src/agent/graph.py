"""
LangGraph multi-hop ReAct agent graph.

Flow:
  START → reason → retrieve → check → (loop: reason | exit: generate) → END

  REASON   — decides next search query + collections
  RETRIEVE — hybrid RAG: HyDE → dense + BM25 → RRF → reranker
  CHECK    — decides if enough retrieved (or max hops reached)
  GENERATE — final answer + citations + conflict warning

Entry point: run_research()
"""

from langgraph.graph import END, StateGraph

from src.agent.nodes import check_node, generate_node, reason_node, retrieve_node
from src.agent.state import LegalResearchState
from src.agent.tools import build_tools
from src.router.guardrail import GuardrailRejection, check_or_raise
from src.router.jurisdiction import get_collections


# ---------------------------------------------------------------------------
# Conditional edge
# ---------------------------------------------------------------------------

def _should_continue(state: LegalResearchState) -> str:
    """Loop back to REASON or exit to GENERATE."""
    if state.get("sufficient") or state["hop_count"] >= state["max_hops"]:
        return "generate"
    return "reason"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph(tools: dict) -> StateGraph:
    graph = StateGraph(LegalResearchState)

    graph.add_node("reason",   reason_node)
    graph.add_node("retrieve", lambda s: retrieve_node(s, tools))
    graph.add_node("check",    check_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("reason")
    graph.add_edge("reason",   "retrieve")
    graph.add_edge("retrieve", "check")
    graph.add_conditional_edges(
        "check",
        _should_continue,
        {"reason": "reason", "generate": "generate"},
    )
    graph.add_edge("generate", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_research(
    question:                str,
    session_id:              str,
    conversation_history:    str = "",
    uploaded_doc_collection: str | None = None,
    max_hops:                int = 4,
) -> LegalResearchState:
    """
    Run the full multi-hop legal research pipeline.

    Steps before the graph:
      1. Legal guardrail — raises GuardrailRejection if not a legal question
      2. Jurisdiction detection — routes to correct ChromaDB collections
      3. Build retrieval tools for those collections
      4. Invoke LangGraph

    Args:
        question:                user's legal question
        session_id:              unique session identifier
        conversation_history:    compressed prior conversation summary
        uploaded_doc_collection: ChromaDB collection name for uploaded PDF (if any)
        max_hops:                maximum retrieval hops (default 4)

    Returns:
        Final LegalResearchState with final_answer, citations, conflict_warning, outcome

    Raises:
        GuardrailRejection: if the question is not legal
    """
    # Step 1 — Guardrail
    check_or_raise(question)

    # Step 2 — Jurisdiction detection
    jurisdiction, _ = get_collections(question)

    # Step 3 — Build tools
    tools = build_tools(
        jurisdiction=jurisdiction,
        uploaded_doc_collection=uploaded_doc_collection,
    )

    # Step 4 — Build + run graph
    app = _build_graph(tools)

    initial_state: LegalResearchState = {
        "question":                question,
        "jurisdiction":            jurisdiction,
        "session_id":              session_id,
        "uploaded_doc_collection": uploaded_doc_collection,
        "hops":                    [],
        "hop_count":               0,
        "max_hops":                max_hops,
        "reasoning_trace":         [],
        "next_query":              question,
        "next_collections":        [],
        "sufficient":              False,
        "conversation_history":    conversation_history,
        "final_answer":            "",
        "citations":               [],
        "conflict_warning":        None,
        "outcome":                 None,
    }

    return app.invoke(initial_state)
