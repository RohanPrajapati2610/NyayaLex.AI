from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os


def get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0,
    )


def hyde_enhance(query: str, llm: ChatGroq) -> str:
    """Generate a hypothetical answer to improve embedding quality (HyDE)."""
    prompt = (
        f"Write a short paragraph that would be a factual legal answer to this question. "
        f"Be specific with statute numbers and case names if possible.\n\nQuestion: {query}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def build_retrieval_tool(collection_name: str, vectorstore, bm25_index, reranker):
    """
    Returns a retrieval function for a given ChromaDB collection.
    Runs: HyDE → dense + BM25 → RRF fusion → cross-encoder reranker → top-5
    """
    from src.vectorstore.hybrid import hybrid_retrieve

    @tool(name=f"search_{collection_name}")
    def retrieve(query: str) -> list[dict]:
        f"""Search the {collection_name} collection using hybrid retrieval (dense + BM25 + reranker)."""
        llm = get_llm()
        hypothetical = hyde_enhance(query, llm)
        chunks = hybrid_retrieve(
            query=query,
            hypothetical=hypothetical,
            vectorstore=vectorstore,
            bm25_index=bm25_index,
            reranker=reranker,
            top_k=5,
        )
        return [c.__dict__ for c in chunks]

    return retrieve
