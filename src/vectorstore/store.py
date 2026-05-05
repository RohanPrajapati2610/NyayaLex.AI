"""
ChromaDB client and collection management.

6 collections — one per data source:
  us_statutes        — all 54 US Code titles
  us_case_law        — 28k SCOTUS opinions (majority/dissent/concurrence)
  us_regulations     — CFR selected titles
  india_statutes     — all Indian central acts
  india_constitution — Constitution of India
  india_case_law     — 5k SC India judgments

Run build_all() once after all ingestion phases complete to embed and
load every chunk into ChromaDB.

Run:
    python -m src.vectorstore.store
"""

import os
from pathlib import Path

import chromadb
import jsonlines
from dotenv import load_dotenv
from tqdm import tqdm

from src.vectorstore.embedder import embed_documents

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHROMA_DIR   = Path("data/vectorstore")
BATCH_SIZE   = 64          # chunks embedded + upserted per batch

COLLECTIONS = {
    "us_statutes":        Path("data/processed/us_statutes.jsonl"),
    "us_case_law":        Path("data/processed/us_case_law.jsonl"),
    "us_regulations":     Path("data/processed/us_regulations.jsonl"),
    "india_statutes":     Path("data/processed/india_statutes.jsonl"),
    "india_constitution": Path("data/processed/india_constitution.jsonl"),
    "india_case_law":     Path("data/processed/india_case_law.jsonl"),
}

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
_client: chromadb.PersistentClient | None = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client


def get_collection(name: str) -> chromadb.Collection:
    """Return (or create) a ChromaDB collection by name."""
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# Build — embed + upsert all chunks from a JSONL file
# ---------------------------------------------------------------------------

def _build_collection(name: str, jsonl_path: Path) -> int:
    """Embed all chunks in jsonl_path and upsert into the named collection."""
    if not jsonl_path.exists():
        print(f"  [SKIP] {jsonl_path} not found — run ingestion first.")
        return 0

    collection = get_collection(name)
    existing   = set(collection.get(include=[])["ids"])

    batch_ids, batch_texts, batch_metas = [], [], []
    total = 0

    with jsonlines.open(jsonl_path) as reader:
        for chunk in tqdm(reader, desc=f"  {name}", unit="chunk"):
            chunk_id = chunk["id"]
            if chunk_id in existing:
                continue

            batch_ids.append(chunk_id)
            batch_texts.append(chunk["text"])
            batch_metas.append(chunk["metadata"])

            if len(batch_ids) >= BATCH_SIZE:
                _upsert_batch(collection, batch_ids, batch_texts, batch_metas)
                total += len(batch_ids)
                batch_ids, batch_texts, batch_metas = [], [], []

    if batch_ids:
        _upsert_batch(collection, batch_ids, batch_texts, batch_metas)
        total += len(batch_ids)

    return total


def _upsert_batch(
    collection: chromadb.Collection,
    ids: list[str],
    texts: list[str],
    metadatas: list[dict],
) -> None:
    embeddings = embed_documents(texts)
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def dense_search(
    collection_name: str,
    query_embedding: list[float],
    n_results: int = 10,
    where: dict | None = None,
) -> list[dict]:
    """
    Dense vector search against a single collection.

    Returns list of dicts with keys: id, text, metadata, score
    """
    collection = get_collection(collection_name)
    kwargs = {
        "query_embeddings": [query_embedding],
        "n_results":        n_results,
        "include":          ["documents", "metadatas", "distances"],
    }
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    hits = []
    for i, doc_id in enumerate(results["ids"][0]):
        hits.append({
            "id":       doc_id,
            "text":     results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score":    1 - results["distances"][0][i],  # cosine distance → similarity
        })
    return hits


def multi_collection_dense_search(
    collection_names: list[str],
    query_embedding: list[float],
    n_results_per: int = 10,
    where: dict | None = None,
) -> list[dict]:
    """Dense search across multiple collections, results merged and sorted by score."""
    all_hits = []
    for name in collection_names:
        all_hits.extend(dense_search(name, query_embedding, n_results_per, where))
    return sorted(all_hits, key=lambda h: h["score"], reverse=True)


# ---------------------------------------------------------------------------
# Entry point — build all 6 collections
# ---------------------------------------------------------------------------

def build_all() -> None:
    """
    Embed and load all 6 collections into ChromaDB.
    Skips chunks already present — safe to re-run (idempotent).
    Expects all processed JSONL files to exist in data/processed/.
    """
    print("Building ChromaDB collections...\n")
    grand_total = 0
    for name, path in COLLECTIONS.items():
        print(f"Collection: {name}")
        n = _build_collection(name, path)
        print(f"  → {n:,} new chunks upserted\n")
        grand_total += n
    print(f"Done. {grand_total:,} total chunks loaded across all collections.")


if __name__ == "__main__":
    build_all()
