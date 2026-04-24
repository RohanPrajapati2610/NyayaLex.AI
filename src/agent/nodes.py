from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.state import LegalResearchState
import json
import os


def get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )


def reason_node(state: LegalResearchState) -> dict:
    """
    REASON node — LLM reads all retrieved chunks so far and decides
    what to search next (or whether to stop).
    """
    llm = get_llm()

    hops_summary = ""
    for i, hop in enumerate(state["hops"]):
        hops_summary += f"\nHop {i+1} retrieved from {hop.get('collection', 'unknown')}:\n{hop.get('content', '')[:300]}...\n"

    system = (
        "You are a legal research agent. Your job is to decide what to search next "
        "to fully answer the user's legal question. Analyse what has already been retrieved "
        "and identify gaps. Output a JSON with keys: "
        "'next_query' (the refined search query), "
        "'collection' (one of: us_statutes, us_case_law, us_regulations, india_statutes, india_constitution, india_case_law, uploaded_doc), "
        "'reasoning' (why this search is needed)."
    )
    human = (
        f"Question: {state['question']}\n\n"
        f"Already retrieved ({len(state['hops'])} chunks):\n{hops_summary or 'Nothing yet.'}\n\n"
        f"Conversation context:\n{state.get('conversation_history', 'None')}\n\n"
        "What should I search next?"
    )

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])

    try:
        parsed = json.loads(response.content)
    except Exception:
        parsed = {"next_query": state["question"], "collection": "us_statutes", "reasoning": "fallback"}

    return {
        "next_query": parsed.get("next_query", state["question"]),
        "reasoning_trace": [parsed.get("reasoning", "")],
    }


def retrieve_node(state: LegalResearchState, tools: dict) -> dict:
    """
    RETRIEVE node — runs hybrid RAG on the collection chosen by REASON node.
    """
    collection = state.get("next_query_collection", state["jurisdiction"])
    tool_fn = tools.get(collection) or tools.get("us_statutes")

    chunks = tool_fn.invoke(state["next_query"])

    return {
        "hops": chunks,
        "hop_count": state["hop_count"] + 1,
    }


def check_node(state: LegalResearchState) -> dict:
    """
    CHECK node — LLM decides if enough information has been retrieved
    to generate a complete answer, or if another hop is needed.
    """
    if state["hop_count"] >= state["max_hops"]:
        return {"sufficient": True}

    llm = get_llm()

    all_content = "\n\n".join(
        f"[{h.get('collection', '')}] {h.get('content', '')[:400]}"
        for h in state["hops"]
    )

    system = (
        "You are a legal research agent. Decide if the retrieved information is sufficient "
        "to give a complete, accurate answer to the question. "
        "Output JSON with key 'sufficient': true or false."
    )
    human = f"Question: {state['question']}\n\nRetrieved so far:\n{all_content}"

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])

    try:
        parsed = json.loads(response.content)
        sufficient = parsed.get("sufficient", False)
    except Exception:
        sufficient = True

    return {"sufficient": sufficient}


def generate_node(state: LegalResearchState) -> dict:
    """
    GENERATE node — synthesises all retrieved chunks into a final answer
    with inline citations, using full conversation memory context.
    """
    llm = get_llm()

    context_blocks = []
    for i, chunk in enumerate(state["hops"]):
        context_blocks.append(
            f"[SOURCE {i+1}] {chunk.get('source', 'Unknown')} "
            f"({chunk.get('collection', '')}):\n{chunk.get('content', '')}"
        )
    context = "\n\n".join(context_blocks)

    system = (
        "You are NyayaLex.AI, an expert legal assistant covering US federal law and Indian law. "
        "Answer the question using ONLY the provided sources. "
        "Cite sources inline as [SOURCE 1], [SOURCE 2], etc. "
        "Be direct, precise, and legally accurate. "
        "If sources conflict, acknowledge the conflict explicitly."
    )
    human = (
        f"Conversation history:\n{state.get('conversation_history', 'None')}\n\n"
        f"Question: {state['question']}\n\n"
        f"Sources:\n{context}"
    )

    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=human)])

    citations = [
        {
            "source": h.get("source", "Unknown"),
            "excerpt": h.get("content", "")[:300],
            "court": h.get("metadata", {}).get("court"),
            "date": h.get("metadata", {}).get("date"),
            "score": h.get("score", 0.0),
            "faithful": True,
        }
        for h in state["hops"]
    ]

    return {
        "final_answer": response.content,
        "citations": citations,
    }
