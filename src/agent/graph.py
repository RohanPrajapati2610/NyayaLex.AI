from langgraph.graph import StateGraph, END
from src.agent.state import LegalResearchState
from src.agent.nodes import reason_node, retrieve_node, check_node, generate_node


def should_continue(state: LegalResearchState) -> str:
    """Edge condition — loop back to REASON or exit to GENERATE."""
    if state.get("sufficient") or state["hop_count"] >= state["max_hops"]:
        return "generate"
    return "reason"


def build_graph(tools: dict) -> StateGraph:
    """
    Build the multi-hop LangGraph reasoning loop.

    Graph flow:
      reason → retrieve → check → (loop back to reason OR exit to generate) → END
    """
    graph = StateGraph(LegalResearchState)

    graph.add_node("reason", reason_node)
    graph.add_node("retrieve", lambda s: retrieve_node(s, tools))
    graph.add_node("check", check_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("reason")

    graph.add_edge("reason", "retrieve")
    graph.add_edge("retrieve", "check")
    graph.add_conditional_edges(
        "check",
        should_continue,
        {
            "reason": "reason",
            "generate": "generate",
        },
    )
    graph.add_edge("generate", END)

    return graph.compile()


def run_research(
    question: str,
    jurisdiction: str,
    session_id: str,
    tools: dict,
    conversation_history: str = "",
    uploaded_doc_collection: str | None = None,
    max_hops: int = 4,
) -> LegalResearchState:
    """Entry point — initialise state and run the graph."""
    app = build_graph(tools)

    initial_state: LegalResearchState = {
        "question": question,
        "jurisdiction": jurisdiction,
        "session_id": session_id,
        "uploaded_doc_collection": uploaded_doc_collection,
        "hops": [],
        "hop_count": 0,
        "max_hops": max_hops,
        "reasoning_trace": [],
        "next_query": question,
        "sufficient": False,
        "conversation_history": conversation_history,
        "final_answer": "",
        "citations": [],
        "conflict_warning": None,
        "outcome": None,
    }

    result = app.invoke(initial_state)
    return result
