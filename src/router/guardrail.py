"""
Legal guardrail — classifies whether a question is legal before
the pipeline runs. Blocks non-legal questions immediately.

Uses Groq zero-shot classification — fast (~200ms), costs nothing.
"""

from src.llm.groq_client import chat
from src.llm.prompts import guardrail_prompt

REJECTION_MESSAGE = (
    "NyayaLex only answers legal questions — statutes, court cases, "
    "legal rights, contracts, constitutional rights, and regulations. "
    "Please ask a legal question."
)


class GuardrailRejection(Exception):
    """Raised when a question is classified as non-legal."""
    pass


def is_legal(question: str) -> bool:
    """
    Returns True if the question is legal in nature, False otherwise.
    Uses Groq zero-shot classification.
    """
    response = chat(
        messages=guardrail_prompt(question),
        temperature=0.0,
        max_tokens=10,
    )
    return response.strip().upper().startswith("LEGAL")


def check_or_raise(question: str) -> None:
    """
    Raises GuardrailRejection if the question is not legal.
    Call this at the start of the pipeline before any retrieval.
    """
    if not is_legal(question):
        raise GuardrailRejection(REJECTION_MESSAGE)
