"""
Retrieval tools for the LangGraph agent.

Each tool wraps hybrid_retrieve() for one or more ChromaDB collections.
HyDE is applied inside every tool call — the hypothetical answer embedding
is used for dense retrieval instead of the raw query embedding.

build_tools() returns a dict: collection_name → callable(query) → list[dict]
"""

from src.llm.groq_client import simple
from src.llm.prompts import hyde_prompt
from src.vectorstore.embedder import embed_query
from src.vectorstore.hybrid import hybrid_retrieve
from src.router.jurisdiction import collections_for_jurisdiction


def _hyde_embed(query: str, jurisdiction: str, conversation_history: str = "") -> list[float]:
    """Generate HyDE hypothetical answer and embed it for dense retrieval."""
    messages  = hyde_prompt(query, jurisdiction, conversation_history)
    # Flatten messages to a single prompt for simple()
    system    = messages[0]["content"]
    user      = messages[1]["content"]
    hypothetical = simple(f"{system}\n\n{user}", temperature=0.3, max_tokens=256)
    return embed_query(hypothetical)


def _make_tool(collection_names: list[str], jurisdiction: str):
    """Returns a retrieval callable for the given collections."""
    def retrieve(query: str, conversation_history: str = "") -> list[dict]:
        query_embedding = _hyde_embed(query, jurisdiction, conversation_history)
        return hybrid_retrieve(
            query=query,
            query_embedding=query_embedding,
            collection_names=collection_names,
        )
    return retrieve


def build_tools(
    jurisdiction: str,
    uploaded_doc_collection: str | None = None,
) -> dict[str, callable]:
    """
    Build retrieval tools for all relevant collections.

    Returns dict: collection_name → retrieval_fn(query, conversation_history) → list[dict]

    Tools built:
      - One tool per individual collection (for targeted REASON node routing)
      - One "all" tool that searches all jurisdiction collections at once
      - One "uploaded_doc" tool if a PDF was uploaded this session
    """
    tools: dict[str, callable] = {}

    all_collections = collections_for_jurisdiction(jurisdiction)

    # Per-collection tools
    for name in all_collections:
        tools[name] = _make_tool([name], jurisdiction)

    # Cross-collection tool
    tools["all"] = _make_tool(all_collections, jurisdiction)

    # Uploaded document tool (per-session ChromaDB collection)
    if uploaded_doc_collection:
        tools["uploaded_doc"] = _make_tool([uploaded_doc_collection], jurisdiction)

    return tools
