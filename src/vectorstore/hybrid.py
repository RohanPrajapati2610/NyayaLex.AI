"""
Hybrid retriever — Dense + BM25 → RRF fusion → Cross-encoder reranker.

Pipeline per retrieval hop:
  1. HyDE: Groq generates a hypothetical answer → embed that
  2. Dense search: BGE embedding against ChromaDB collections
  3. BM25 search: keyword search against BM25 indexes
  4. RRF fusion: Reciprocal Rank Fusion merges both ranked lists
  5. Cross-encoder reranker: reranks top-N with a neural model
  6. Returns top-K chunks with metadata and scores
"""

from FlagEmbedding import FlagReranker

from src.vectorstore.bm25_index import multi_collection_bm25_search
from src.vectorstore.store import multi_collection_dense_search

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RERANKER_MODEL  = "BAAI/bge-reranker-base"   # runs on CPU, free
RRF_K           = 60                          # RRF constant (standard value)
DENSE_N         = 10                          # candidates per collection from dense
BM25_N          = 10                          # candidates per collection from BM25
RERANK_TOP_N    = 10                          # candidates passed to reranker
FINAL_TOP_K     = 5                           # chunks returned after reranking

_reranker: FlagReranker | None = None


def _get_reranker() -> FlagReranker:
    global _reranker
    if _reranker is None:
        _reranker = FlagReranker(RERANKER_MODEL, use_fp16=False)
    return _reranker


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense_hits: list[dict],
    bm25_hits: list[dict],
    k: int = RRF_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion.

    Score(d) = Σ 1 / (k + rank_i(d))  for each ranked list i

    Deduplicates by chunk ID and returns sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    chunk_map:  dict[str, dict]  = {}

    for rank, hit in enumerate(dense_hits, start=1):
        cid = hit["id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map[cid]  = hit

    for rank, hit in enumerate(bm25_hits, start=1):
        cid = hit["id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_map.setdefault(cid, hit)

    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {**chunk_map[cid], "rrf_score": score}
        for cid, score in fused
    ]


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------

def _rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """
    Cross-encoder reranker — scores each (query, chunk) pair with a neural model.
    Returns top_k chunks sorted by reranker score descending.
    """
    reranker = _get_reranker()
    pairs    = [[query, c["text"]] for c in candidates]
    scores   = reranker.compute_score(pairs, normalize=True)

    for i, chunk in enumerate(candidates):
        chunk["reranker_score"] = float(scores[i])

    return sorted(candidates, key=lambda c: c["reranker_score"], reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_retrieve(
    query: str,
    query_embedding: list[float],
    collection_names: list[str],
    dense_n: int  = DENSE_N,
    bm25_n: int   = BM25_N,
    rerank_n: int = RERANK_TOP_N,
    final_k: int  = FINAL_TOP_K,
) -> list[dict]:
    """
    Full hybrid retrieval pipeline for one hop.

    Args:
        query:            raw query string (used for BM25 + reranker)
        query_embedding:  pre-computed embedding (HyDE or raw query)
        collection_names: which ChromaDB/BM25 collections to search
        dense_n:          candidates per collection from dense search
        bm25_n:           candidates per collection from BM25
        rerank_n:         top-N passed to cross-encoder
        final_k:          final chunks returned after reranking

    Returns:
        list of up to final_k chunks, each with keys:
        id, text, metadata, rrf_score, reranker_score
    """
    # Step 1 — Dense search
    dense_hits = multi_collection_dense_search(
        collection_names, query_embedding, n_results_per=dense_n
    )

    # Step 2 — BM25 search
    bm25_hits = multi_collection_bm25_search(
        collection_names, query, n_results_per=bm25_n
    )

    # Step 3 — RRF fusion
    fused = _rrf_fuse(dense_hits, bm25_hits)

    # Step 4 — Take top rerank_n for cross-encoder (avoid scoring all candidates)
    candidates = fused[:rerank_n]

    # Step 5 — Cross-encoder rerank
    reranked = _rerank(query, candidates, top_k=final_k)

    return reranked
