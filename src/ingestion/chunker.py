"""
Shared chunker for all NyayaLex.AI ingestion phases.

Splits text into token-aware, sentence-safe, overlapping chunks.
Every chunk gets metadata attached so ChromaDB can filter by
jurisdiction, source, title, section, etc.
"""

import re
import uuid
import tiktoken

TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 400
OVERLAP_TOKENS = 50


def _count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences. Keeps sentence boundaries intact."""
    text = re.sub(r"\s+", " ", text).strip()
    # Split on period/exclamation/question followed by space + capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z(\"'])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, metadata: dict, max_tokens: int = MAX_TOKENS, overlap_tokens: int = OVERLAP_TOKENS) -> list[dict]:
    """
    Split text into overlapping chunks, each tagged with metadata.

    Args:
        text:          cleaned section / opinion text
        metadata:      dict with source, jurisdiction, citation, etc.
        max_tokens:    maximum tokens per chunk (default 400)
        overlap_tokens: tokens carried over between chunks for context continuity

    Returns:
        list of dicts — each with "id", "text", "metadata"
    """
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks = []
    current_sentences: list[str] = []
    current_tokens = 0
    overlap_buffer: list[str] = []  # tail sentences carried into next chunk

    for sentence in sentences:
        sentence_tokens = _count_tokens(sentence)

        # Single sentence longer than max — add as its own chunk, split by words
        if sentence_tokens > max_tokens:
            if current_sentences:
                chunks.append(_make_chunk(current_sentences, metadata, len(chunks)))
                overlap_buffer = _tail_sentences(current_sentences, overlap_tokens)
                current_sentences = list(overlap_buffer)
                current_tokens = sum(_count_tokens(s) for s in current_sentences)

            # Forcibly split the long sentence by words
            for word_chunk in _split_long_sentence(sentence, max_tokens):
                chunks.append(_make_chunk([word_chunk], metadata, len(chunks)))
            overlap_buffer = []
            current_sentences = []
            current_tokens = 0
            continue

        if current_tokens + sentence_tokens > max_tokens:
            # Flush current chunk
            if current_sentences:
                chunks.append(_make_chunk(current_sentences, metadata, len(chunks)))
                overlap_buffer = _tail_sentences(current_sentences, overlap_tokens)
            # Start new chunk with overlap
            current_sentences = list(overlap_buffer) + [sentence]
            current_tokens = sum(_count_tokens(s) for s in current_sentences)
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    # Flush remaining sentences
    if current_sentences:
        chunks.append(_make_chunk(current_sentences, metadata, len(chunks)))

    # Stamp total_chunks on all chunks now that we know the count
    total = len(chunks)
    for chunk in chunks:
        chunk["metadata"]["total_chunks"] = total

    return chunks


def _make_chunk(sentences: list[str], metadata: dict, index: int) -> dict:
    text = " ".join(sentences)
    chunk_metadata = {
        **metadata,
        "chunk_index": index,
        "token_count": _count_tokens(text),
    }
    chunk_id = f"{metadata.get('source', 'doc')}_{metadata.get('section_num', 'x')}_{index}_{uuid.uuid4().hex[:6]}"
    return {"id": chunk_id, "text": text, "metadata": chunk_metadata}


def _tail_sentences(sentences: list[str], overlap_tokens: int) -> list[str]:
    """Return the tail sentences that fit within overlap_tokens (for context carry-over)."""
    tail: list[str] = []
    token_count = 0
    for sentence in reversed(sentences):
        t = _count_tokens(sentence)
        if token_count + t > overlap_tokens:
            break
        tail.insert(0, sentence)
        token_count += t
    return tail


def _split_long_sentence(sentence: str, max_tokens: int) -> list[str]:
    """Word-level split for a single sentence that exceeds max_tokens."""
    words = sentence.split()
    chunks = []
    current_words: list[str] = []
    current_tokens = 0
    for word in words:
        word_tokens = _count_tokens(word)
        if current_tokens + word_tokens > max_tokens and current_words:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_tokens = word_tokens
        else:
            current_words.append(word)
            current_tokens += word_tokens
    if current_words:
        chunks.append(" ".join(current_words))
    return chunks
