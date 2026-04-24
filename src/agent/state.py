from typing import TypedDict, Annotated
import operator


class RetrievedChunk(TypedDict):
    content: str
    source: str
    collection: str
    score: float
    metadata: dict


class CitationResult(TypedDict):
    source: str
    excerpt: str
    court: str | None
    date: str | None
    score: float
    faithful: bool


class OutcomePrediction(TypedDict):
    verdict: str
    confidence: float
    label: str


class LegalResearchState(TypedDict):
    # Input
    question: str
    jurisdiction: str                              # "us" | "india" | "both"
    session_id: str
    uploaded_doc_collection: str | None            # per-session ChromaDB collection if PDF uploaded

    # Multi-hop tracking
    hops: Annotated[list[dict], operator.add]      # accumulated retrieved chunks across hops
    hop_count: int
    max_hops: int                                  # default 4
    reasoning_trace: Annotated[list[str], operator.add]  # LLM reasoning at each hop
    next_query: str                                # refined query for next retrieval hop
    sufficient: bool                               # CHECK node sets this to exit the loop

    # Memory
    conversation_history: str                      # compressed by ConversationSummaryMemory

    # Output
    final_answer: str
    citations: list[CitationResult]
    conflict_warning: str | None
    outcome: OutcomePrediction | None
