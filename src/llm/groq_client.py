"""
Groq API client wrapper.

Single entry point for all LLM calls in the pipeline.
Handles retries, timeouts, and streaming.

Model: llama-3.3-70b-versatile (free tier — 6,000 tokens/min, 100,000 tokens/day)
"""

import os
import time

from groq import Groq
from dotenv import load_dotenv

load_dotenv()

MODEL       = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_RETRIES = 4
RETRY_WAIT  = 2  # seconds, doubled on each retry

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at console.groq.com "
                "and add it to .env"
            )
        _client = Groq(api_key=api_key)
    return _client


def chat(
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 1024,
    stream: bool = False,
) -> str:
    """
    Send a chat completion request to Groq.

    Args:
        messages:    list of {"role": "system"|"user"|"assistant", "content": str}
        temperature: 0.0 for deterministic legal answers
        max_tokens:  cap on response length
        stream:      if True, returns a generator of token strings

    Returns:
        full response string (or generator if stream=True)
    """
    client = _get_client()

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
            )
            if stream:
                return _stream_generator(resp)
            return resp.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                wait = RETRY_WAIT * (2 ** attempt)
                time.sleep(wait)
                continue
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_WAIT)

    raise RuntimeError("Groq API failed after max retries")


def _stream_generator(resp):
    """Yield token strings from a Groq streaming response."""
    for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def simple(prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """Convenience wrapper for single-turn prompts."""
    return chat(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
