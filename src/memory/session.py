"""
Multi-turn conversation memory — ConversationSummaryMemory pattern.

How it works:
  - Every Q&A exchange is appended to a per-session exchange buffer
  - When buffer exceeds MAX_EXCHANGES, Groq compresses it into a
    running summary (so context never grows unbounded)
  - The compressed summary is injected into every LangGraph node
    that needs prior context (reason, generate)

Storage: in-memory dict keyed by session_id.
  - Sessions are lost on server restart (acceptable for a demo)
  - For production: swap _store with Redis or a DB

Session lifecycle:
  create_session()  → returns session_id
  add_exchange()    → appends Q+A, compresses if needed
  get_summary()     → returns current compressed summary for injection
  clear_session()   → resets session
"""

import uuid
from dataclasses import dataclass, field

from src.llm.groq_client import chat
from src.llm.prompts import summarise_conversation_prompt

MAX_EXCHANGES    = 4     # compress after this many Q&A pairs
MAX_SUMMARY_AGE  = 20    # hard cap — always compress after 20 exchanges


@dataclass
class Session:
    session_id:  str
    summary:     str = ""                    # compressed running summary
    exchanges:   list[dict] = field(default_factory=list)   # recent raw exchanges
    total_turns: int = 0


# In-memory store: session_id → Session
_store: dict[str, Session] = {}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def create_session() -> str:
    """Create a new session and return its ID."""
    sid = str(uuid.uuid4())
    _store[sid] = Session(session_id=sid)
    return sid


def get_session(session_id: str) -> Session | None:
    return _store.get(session_id)


def clear_session(session_id: str) -> None:
    if session_id in _store:
        del _store[session_id]


# ---------------------------------------------------------------------------
# Memory operations
# ---------------------------------------------------------------------------

def add_exchange(session_id: str, question: str, answer: str) -> None:
    """
    Add a Q&A exchange to the session.
    Compresses old exchanges into summary when buffer exceeds MAX_EXCHANGES.
    """
    if session_id not in _store:
        _store[session_id] = Session(session_id=session_id)

    session = _store[session_id]

    session.exchanges.append({"role": "user",      "content": question})
    session.exchanges.append({"role": "assistant",  "content": answer})
    session.total_turns += 1

    # Compress when buffer is full
    if len(session.exchanges) >= MAX_EXCHANGES * 2:
        _compress(session)


def get_summary(session_id: str) -> str:
    """
    Return the current memory context for injection into the agent.
    Combines compressed summary + any recent raw exchanges not yet compressed.
    """
    session = _store.get(session_id)
    if not session:
        return ""

    parts = []
    if session.summary:
        parts.append(f"Prior conversation summary:\n{session.summary}")

    if session.exchanges:
        recent = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
            for m in session.exchanges[-4:]   # last 2 Q&A pairs uncompressed
        )
        parts.append(f"Recent exchanges:\n{recent}")

    return "\n\n".join(parts)


def get_history(session_id: str) -> list[dict]:
    """Return raw exchange list for a session (for evaluation/logging)."""
    session = _store.get(session_id)
    return session.exchanges if session else []


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def _compress(session: Session) -> None:
    """
    Compress session.exchanges into session.summary using Groq.
    Keeps only the last 2 exchanges (1 Q&A pair) as recent raw context.
    """
    # Keep the most recent 2 messages uncompressed
    to_compress   = session.exchanges[:-2]
    session.exchanges = session.exchanges[-2:]

    if not to_compress:
        return

    response = chat(
        messages=summarise_conversation_prompt(
            previous_summary=session.summary,
            new_exchanges=to_compress,
        ),
        temperature=0.0,
        max_tokens=200,
    )

    session.summary = response.strip()
