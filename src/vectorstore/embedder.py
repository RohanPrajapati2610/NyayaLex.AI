"""
BGE embedding model wrapper.

Uses BAAI/bge-large-en-v1.5 — runs on CPU, no GPU needed.
BGE requires a query prefix on queries (not on documents).
"""

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
QUERY_PREFIX    = "Represent this sentence for searching relevant passages: "

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of document chunks — no prefix applied."""
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()


def embed_query(query: str) -> list[float]:
    """Embed a single query — BGE query prefix applied."""
    model = _get_model()
    prefixed = QUERY_PREFIX + query
    return model.encode(prefixed, normalize_embeddings=True, show_progress_bar=False).tolist()
