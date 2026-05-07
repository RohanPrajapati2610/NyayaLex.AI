"""
All prompt templates for NyayaLex.AI pipeline.

Each function returns a list of message dicts ready for groq_client.chat().
Temperature guidance is noted per prompt.
"""


# ---------------------------------------------------------------------------
# 1. Legal Guardrail
# ---------------------------------------------------------------------------

def guardrail_prompt(question: str) -> list[dict]:
    """
    Classifies whether a question is legal in nature.
    Use temperature=0.0 — binary classification.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a legal topic classifier. "
                "Reply with exactly one word: LEGAL or NOT_LEGAL. "
                "A question is LEGAL if it relates to laws, statutes, court cases, "
                "legal rights, contracts, criminal procedure, constitutional rights, "
                "regulations, or legal advice. "
                "Reply NOT_LEGAL for cooking, sports, weather, general knowledge, "
                "or anything unrelated to law."
            ),
        },
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# 2. Jurisdiction Router
# ---------------------------------------------------------------------------

def jurisdiction_prompt(question: str) -> list[dict]:
    """
    Routes the question to a jurisdiction.
    Returns one of: US, INDIA, BOTH
    Use temperature=0.0.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a legal jurisdiction classifier. "
                "Classify the question into one of three jurisdictions:\n"
                "  US    — question is about US federal law, SCOTUS, US statutes or regulations\n"
                "  INDIA — question is about Indian law, IPC/BNS, Indian Constitution, SC India\n"
                "  BOTH  — question involves both jurisdictions or is ambiguous\n\n"
                "Reply with exactly one word: US, INDIA, or BOTH."
            ),
        },
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# 3. HyDE — Hypothetical Document Embedding
# ---------------------------------------------------------------------------

def hyde_prompt(question: str, jurisdiction: str, conversation_summary: str = "") -> list[dict]:
    """
    Generates a hypothetical legal answer to improve embedding quality.
    The embedding of this hypothetical is used for retrieval — not shown to user.
    Use temperature=0.3 — slight variation helps retrieval diversity.
    """
    context = ""
    if conversation_summary:
        context = f"\nConversation context:\n{conversation_summary}\n"

    jurisdiction_hint = {
        "US":    "US federal law, US Code, SCOTUS case law, and CFR regulations",
        "INDIA": "Indian law, BNS/BNSS, Indian Constitution, and Supreme Court of India judgments",
        "BOTH":  "both US federal law and Indian law",
    }.get(jurisdiction, "applicable law")

    return [
        {
            "role": "system",
            "content": (
                "You are a legal expert. Write a concise, authoritative paragraph "
                f"answering the question using {jurisdiction_hint}. "
                "Include specific section numbers, act names, or case citations where relevant. "
                "This is used internally for document retrieval — be specific and legal in tone."
                f"{context}"
            ),
        },
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# 4. ReAct REASON node
# ---------------------------------------------------------------------------

def reason_prompt(
    question: str,
    hops_so_far: list[dict],
    conversation_summary: str = "",
) -> list[dict]:
    """
    Decides what to search next given what has been retrieved so far.
    Returns a JSON with: {"search_query": "...", "collections": [...], "reasoning": "..."}
    Use temperature=0.0.
    """
    hop_summaries = ""
    if hops_so_far:
        parts = []
        for i, hop in enumerate(hops_so_far, 1):
            sources = ", ".join(
                f"{c.get('metadata', {}).get('citation', 'Unknown')}"
                for c in hop.get("chunks", [])
            )
            parts.append(f"Hop {i} searched: '{hop['query']}' → found: {sources or 'nothing relevant'}")
        hop_summaries = "\n".join(parts)

    context = f"\nConversation context:\n{conversation_summary}\n" if conversation_summary else ""

    return [
        {
            "role": "system",
            "content": (
                "You are a legal research agent deciding what to search next.\n"
                "Available collections: us_statutes, us_case_law, us_regulations, "
                "india_statutes, india_constitution, india_case_law\n\n"
                "Respond with ONLY valid JSON in this format:\n"
                '{"search_query": "specific legal search query", '
                '"collections": ["collection1", "collection2"], '
                '"reasoning": "why this search"}'
                f"{context}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Retrieved so far:\n{hop_summaries or 'Nothing yet — this is the first search.'}\n\n"
                "What should I search next to best answer this question?"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# 5. ReAct CHECK node
# ---------------------------------------------------------------------------

def check_prompt(question: str, hops_so_far: list[dict]) -> list[dict]:
    """
    Decides if enough information has been retrieved to answer the question.
    Returns JSON: {"enough": true/false, "missing": "what is still needed"}
    Use temperature=0.0.
    """
    retrieved_text = ""
    for i, hop in enumerate(hops_so_far, 1):
        for chunk in hop.get("chunks", []):
            citation = chunk.get("metadata", {}).get("citation", "Unknown")
            preview  = chunk.get("text", "")[:200]
            retrieved_text += f"\n[Hop {i} — {citation}]: {preview}..."

    return [
        {
            "role": "system",
            "content": (
                "You are a legal research evaluator. "
                "Decide if the retrieved information is sufficient to answer the question fully.\n"
                "Respond with ONLY valid JSON:\n"
                '{"enough": true, "missing": ""} or {"enough": false, "missing": "what is needed"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Retrieved information:{retrieved_text}\n\n"
                "Is this enough to answer the question fully and accurately?"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# 6. GENERATE — Final answer
# ---------------------------------------------------------------------------

def generate_prompt(
    question: str,
    retrieved_chunks: list[dict],
    jurisdiction: str,
    conversation_summary: str = "",
) -> list[dict]:
    """
    Generates the final answer with inline citations.
    Use temperature=0.0 for legal accuracy.
    """
    sources_block = ""
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta     = chunk.get("metadata", {})
        citation = meta.get("citation", f"Source {i}")
        text     = chunk.get("text", "")[:500]
        sources_block += f"\n[{i}] {citation}:\n{text}\n"

    context = f"\nConversation context:\n{conversation_summary}\n" if conversation_summary else ""

    jurisdiction_instruction = {
        "US":    "Answer using US federal law. Cite US Code sections and SCOTUS cases.",
        "INDIA": "Answer using Indian law. Cite BNS/BNSS sections and Supreme Court of India cases.",
        "BOTH":  "Answer using both US and Indian law where relevant. Clearly label which jurisdiction each point applies to.",
    }.get(jurisdiction, "Answer based on the retrieved sources.")

    return [
        {
            "role": "system",
            "content": (
                "You are NyayaLex, an expert legal research assistant. "
                f"{jurisdiction_instruction}\n\n"
                "Rules:\n"
                "- Cite sources using [1], [2], etc. inline in your answer\n"
                "- Be direct and precise — this is legal information, not legal advice\n"
                "- If sources conflict, acknowledge the conflict explicitly\n"
                "- Never invent citations — only cite what is in the provided sources\n"
                "- End with: 'Note: This is legal information, not legal advice. "
                "Consult a qualified lawyer for your specific situation.'"
                f"{context}"
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nSources:{sources_block}\n\nPlease answer the question.",
        },
    ]


# ---------------------------------------------------------------------------
# 7. Conversation memory compression
# ---------------------------------------------------------------------------

def summarise_conversation_prompt(
    previous_summary: str,
    new_exchanges: list[dict],
) -> list[dict]:
    """
    Compresses conversation history into a running summary.
    Used by ConversationSummaryMemory to keep context compact.
    Use temperature=0.0.
    """
    exchanges_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in new_exchanges
    )

    return [
        {
            "role": "system",
            "content": (
                "Summarise the following legal conversation exchanges into a concise paragraph. "
                "Preserve: key legal questions asked, jurisdictions discussed, statutes or cases cited, "
                "and any conclusions reached. Be brief — under 150 words."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Previous summary:\n{previous_summary or 'None — first exchange.'}\n\n"
                f"New exchanges:\n{exchanges_text}\n\n"
                "Updated summary:"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# 8. Conflict detection
# ---------------------------------------------------------------------------

def conflict_detection_prompt(question: str, chunks: list[dict]) -> list[dict]:
    """
    Detects if any retrieved sources contradict each other.
    Returns JSON: {"conflict": true/false, "explanation": "..."}
    Use temperature=0.0.
    """
    sources_text = ""
    for i, chunk in enumerate(chunks, 1):
        citation = chunk.get("metadata", {}).get("citation", f"Source {i}")
        text     = chunk.get("text", "")[:300]
        sources_text += f"\n[{i}] {citation}: {text}\n"

    return [
        {
            "role": "system",
            "content": (
                "You are a legal conflict analyser. "
                "Check if any of the provided legal sources contradict each other "
                "on the question asked.\n"
                "Respond with ONLY valid JSON:\n"
                '{"conflict": false, "explanation": ""} or '
                '{"conflict": true, "explanation": "Source [1] says X but Source [2] says Y..."}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\nSources:{sources_text}\n\n"
                "Do any of these sources contradict each other?"
            ),
        },
    ]
