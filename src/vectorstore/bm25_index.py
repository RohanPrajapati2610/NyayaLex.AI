"""
BM25 sparse index — one per collection.

Built from the same JSONL files used by ChromaDB.
Persisted to data/vectorstore/bm25_{collection}.pkl so it
loads instantly on restart without re-indexing.

Run:
    python -m src.vectorstore.bm25_index
"""

import pickle
from pathlib import Path

import jsonlines
from rank_bm25 import BM25Okapi
from tqdm import tqdm

INDEX_DIR = Path("data/vectorstore/bm25")

COLLECTIONS = {
    "us_statutes":        Path("data/processed/us_statutes.jsonl"),
    "us_case_law":        Path("data/processed/us_case_law.jsonl"),
    "us_regulations":     Path("data/processed/us_regulations.jsonl"),
    "india_statutes":     Path("data/processed/india_statutes.jsonl"),
    "india_constitution": Path("data/processed/india_constitution.jsonl"),
    "india_case_law":     Path("data/processed/india_case_law.jsonl"),
}

# In-memory cache: collection_name → (BM25Okapi, list[dict])
_indexes: dict[str, tuple[BM25Okapi, list[dict]]] = {}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_index(name: str, jsonl_path: Path) -> None:
    """Build and persist BM25 index for one collection."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    out = INDEX_DIR / f"{name}.pkl"

    if not jsonl_path.exists():
        print(f"  [SKIP] {jsonl_path} not found — run ingestion first.")
        return

    print(f"  Building BM25 index: {name}")
    corpus_tokens = []
    chunks        = []

    with jsonlines.open(jsonl_path) as reader:
        for chunk in tqdm(reader, desc=f"    tokenising {name}", unit="chunk"):
            tokens = chunk["text"].lower().split()
            corpus_tokens.append(tokens)
            chunks.append({"id": chunk["id"], "text": chunk["text"], "metadata": chunk["metadata"]})

    index = BM25Okapi(corpus_tokens)
    with open(out, "wb") as f:
        pickle.dump({"index": index, "chunks": chunks}, f)

    print(f"  → {len(chunks):,} chunks indexed, saved to {out}\n")


def build_all() -> None:
    print("Building BM25 indexes...\n")
    for name, path in COLLECTIONS.items():
        build_index(name, path)
    print("Done.")


# ---------------------------------------------------------------------------
# Load + Search
# ---------------------------------------------------------------------------

def _load(name: str) -> tuple[BM25Okapi, list[dict]]:
    if name not in _indexes:
        pkl = INDEX_DIR / f"{name}.pkl"
        if not pkl.exists():
            raise FileNotFoundError(
                f"BM25 index for '{name}' not found at {pkl}. "
                "Run: python -m src.vectorstore.bm25_index"
            )
        with open(pkl, "rb") as f:
            data = pickle.load(f)
        _indexes[name] = (data["index"], data["chunks"])
    return _indexes[name]


def bm25_search(name: str, query: str, n_results: int = 10) -> list[dict]:
    """
    BM25 keyword search against one collection.

    Returns list of dicts: id, text, metadata, score (BM25 raw score)
    """
    index, chunks = _load(name)
    tokens = query.lower().split()
    scores = index.get_scores(tokens)

    ranked = sorted(
        [(score, chunks[i]) for i, score in enumerate(scores)],
        key=lambda x: x[0],
        reverse=True,
    )[:n_results]

    return [
        {
            "id":       chunk["id"],
            "text":     chunk["text"],
            "metadata": chunk["metadata"],
            "score":    float(score),
        }
        for score, chunk in ranked
    ]


def multi_collection_bm25_search(
    names: list[str],
    query: str,
    n_results_per: int = 10,
) -> list[dict]:
    """BM25 search across multiple collections. Raw scores are NOT comparable
    across collections — use RRF fusion downstream."""
    all_hits = []
    for name in names:
        all_hits.extend(bm25_search(name, query, n_results_per))
    return all_hits


if __name__ == "__main__":
    build_all()
