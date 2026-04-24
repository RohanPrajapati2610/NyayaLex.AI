# NyayaLex.AI — Legal AI Research System

A production-grade Legal AI system covering **US federal law** and **Indian law** — combining multi-hop agentic RAG, multi-turn memory, case outcome prediction, document analysis, jurisdiction-aware routing, and voice I/O over real government legal data.

**Headline result:** Hallucination rate reduced from **59% → 17%** versus a base LLM, verified across ROUGE, BERTScore, and RAGAS metrics.

---

## What Makes This Different From A Basic Legal Chatbot

A basic legal chatbot sends your question to an LLM and returns an answer. That is an API wrapper.

NyayaLex.AI runs **7 AI models per query** across a multi-hop reasoning loop:

```
User question / voice / uploaded PDF
  ↓
Model 1 — Groq LLaMA-3.3-70B      : Legal guardrail (is this a legal question?)
  ↓
Model 2 — Jurisdiction Router      : routes to US / India / both collections
  ↓
  LangGraph ReAct Loop (up to 4 hops):
    Model 3 — Groq LLaMA-3.3-70B  : REASON — decides what to search next
    Model 4 — BGE-large-en-v1.5   : embeds HyDE hypothetical answer (335M params)
              BM25                 : keyword search in parallel
              RRF fusion           : merges dense + sparse results
    Model 5 — Cross-encoder        : reranks top-10 → top-5 (neural second pass)
              CHECK                : enough info? loop again or proceed?
  ↓
Model 6 — Groq LLaMA-3.3-70B      : GENERATE final answer with all hop context
  ↓
Model 7 — NLI model                : verifies every citation is faithful
           Conflict detector       : flags contradicting sources
  ↓
Model 8 — legal-bert-base-uncased  : predicts case outcome + confidence % (110M params)
  ↓
FastAPI → Next.js frontend (Vercel) + browser TTS speaks answer aloud
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER — real government documents, ingested once      │
│                                                             │
│  US:     All 54 US Code titles  (uscode.house.gov)          │
│          28,000 SCOTUS opinions (CourtListener API)         │
│          Code of Federal Regulations — selected titles      │
│                                                             │
│  India:  All central acts — BNS, BNSS, BSA, Contract,       │
│          Companies Act, etc.  (indiacode.nic.in API)        │
│          Constitution of India (full text)                  │
│          Supreme Court of India judgments (Indian Kanoon)   │
│                                                             │
│  Stored in 6 ChromaDB collections on HuggingFace Hub       │
│  (~4GB total, within 50GB free limit)                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  RETRIEVAL LAYER                                            │
│                                                             │
│  HyDE: LLM generates hypothetical answer → embed that       │
│  Dense: BGE-large-en-v1.5 vector search in ChromaDB         │
│  Sparse: BM25 keyword search                                │
│  Fusion: Reciprocal Rank Fusion merges both                 │
│  Rerank: Cross-encoder neural reranker top-10 → top-5       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  AGENT LAYER — LangGraph multi-hop ReAct loop               │
│                                                             │
│  REASON → RETRIEVE → CHECK → (loop up to 4 hops)           │
│  Each hop refines the query based on what was retrieved     │
│  Simple queries: 1-2 hops (~4s)                             │
│  Complex multi-statute queries: 3-4 hops (~10s)             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  QUALITY LAYER                                              │
│                                                             │
│  NLI faithfulness: does cited source actually support claim?│
│  Conflict detector: do any two sources contradict each other│
│  Outcome predictor: legal-bert predicts ruling + confidence │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  INTERFACE LAYER                                            │
│                                                             │
│  FastAPI backend  → HuggingFace Spaces (free)               │
│  Next.js frontend → Vercel (free)                           │
│  Voice: browser Web Speech API (STT + TTS, no backend)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Coverage

### United States — All 54 US Code Titles

| Title | Subject |
|---|---|
| 1 | General Provisions + Constitution |
| 2 | The Congress |
| 3 | The President |
| 4 | Flag, Seal, Government, States |
| 5 | Government Organization & Employees |
| 6 | Domestic Security (Homeland Security) |
| 7 | Agriculture |
| 8 | Aliens & Nationality (Immigration) |
| 9 | Arbitration |
| 10 | Armed Forces (Military Law) |
| 11 | Bankruptcy |
| 12 | Banks & Banking |
| 13 | Census |
| 14 | Coast Guard |
| 15 | Commerce & Trade |
| 16 | Conservation |
| 17 | Copyrights |
| 18 | Crimes & Criminal Procedure |
| 19 | Customs Duties |
| 20 | Education |
| 21 | Food & Drugs (FDA) |
| 22 | Foreign Relations |
| 23 | Highways |
| 24 | Hospitals & Asylums |
| 25 | Indians (Native American Law) |
| 26 | Internal Revenue Code (Tax) |
| 27 | Intoxicating Liquors |
| 28 | Judiciary & Judicial Procedure |
| 29 | Labor |
| 30 | Mineral Lands & Mining |
| 31 | Money & Finance |
| 32 | National Guard |
| 33 | Navigation & Navigable Waters |
| 34 | Crime Control & Law Enforcement |
| 35 | Patents |
| 36 | Patriotic & National Observances |
| 37 | Pay & Allowances of Uniformed Services |
| 38 | Veterans' Benefits |
| 39 | Postal Service |
| 40 | Public Buildings & Works |
| 41 | Public Contracts |
| 42 | Public Health & Welfare (Civil Rights) |
| 43 | Public Lands |
| 44 | Public Printing & Documents |
| 45 | Railroads |
| 46 | Shipping |
| 47 | Telecommunications |
| 48 | Territories & Insular Possessions |
| 49 | Transportation |
| 50 | War & National Defense |
| 51 | National & Commercial Space Programs |
| 52 | Voting & Elections |
| 53 | Reserved — not enacted |
| 54 | National Park Service |

Plus: **28,000 SCOTUS opinions** (majority + dissenting + concurring opinions) via CourtListener API.
Plus: **Code of Federal Regulations** — selected titles (21 Food/Drug, 26 Tax, 12 Banking).

### India — Full Central Law Coverage

| Source | Content |
|---|---|
| India Code API (indiacode.nic.in) | All central acts — BNS 2023, BNSS 2023, BSA 2023, Indian Contract Act, Companies Act 2013, and all others |
| Constitution of India | Full text — all articles, parts, schedules |
| Indian Kanoon API | Supreme Court of India judgments (~5,000 cases, full text) |

---

## AI Features

| Feature | How |
|---|---|
| Multi-hop reasoning | LangGraph ReAct loop — each hop refines based on prior results |
| Multi-turn memory | ConversationSummaryMemory — LLM compresses history, maintains context |
| Jurisdiction routing | Auto-detects US / India / both from question |
| Conflict detection | NLI across retrieved sources — flags contradicting statutes |
| Citation faithfulness | NLI entailment — verifies every cited source supports the claim |
| Case outcome prediction | legal-bert-base-uncased — predicts ruling + confidence % |
| Document upload | PyMuPDF parses user's PDF, per-session ChromaDB collection, targeted RAG |
| Voice I/O | Browser Web Speech API — speaks question, hears answer |
| Legal guardrail | Groq zero-shot — rejects non-legal questions politely |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — LLaMA-3.3-70B (free tier) |
| Agent | LangGraph 0.2.x — multi-hop ReAct loop |
| Memory | LangChain ConversationSummaryMemory |
| Outcome Predictor | nlpaueb/legal-bert-base-uncased (HuggingFace, CPU) |
| Embeddings | BAAI/bge-large-en-v1.5 |
| Vector DB | ChromaDB 0.5.x — 6 collections (US statutes, US case law, US regulations, India statutes, India constitution, India case law) |
| Sparse Retrieval | BM25 (rank-bm25) |
| Reranker | FlagEmbedding cross-encoder |
| Document Parsing | PyMuPDF |
| Framework | LangChain 0.3.x |
| Evaluation | RAGAS 0.2.x + rouge-score + bert-score |
| Experiment Tracking | MLflow 2.18.x |
| API | FastAPI 0.115.x |
| Frontend | Next.js 15 (Vercel, free) |
| Voice | Browser Web Speech API (STT + TTS — no backend required) |
| HF Hub | huggingface_hub 0.26.x — pre-built ChromaDB download on Spaces startup |
| Deploy | Backend: HuggingFace Spaces · Frontend: Vercel — both free |

---

## Target Metrics

| Metric | Base LLM | RAG + Full Pipeline |
|---|---|---|
| ROUGE-L | ~0.18 | ~0.53 |
| BERTScore F1 | ~0.71 | ~0.86 |
| Faithfulness (RAGAS) | ~0.41 | ~0.83 |
| Hallucination Rate | ~59% | ~17% |

---

## Response Time

| Query type | Written answer | Voice starts |
|---|---|---|
| Simple (1 hop) | ~3–4 seconds | ~3–4 seconds |
| Medium (2 hops) | ~5–6 seconds | ~5–6 seconds |
| Complex (3–4 hops) | ~8–11 seconds | ~8–11 seconds |

Voice (TTS) begins the moment the written answer appears. Browser SpeechSynthesis is near-instant.

---

## Data Storage Architecture

```
Your laptop (one time only):
  Downloads raw XML/JSON from government APIs (~7GB total)
  Processes → chunks → embeds → builds ChromaDB (~4GB)
  Uploads ChromaDB to HuggingFace Hub Dataset repo
  Raw files deleted — never committed to git

GitHub (code only):
  src/, frontend/, scripts/, docker-compose.yml, requirements.txt
  .gitignore blocks all of data/

HuggingFace Hub — Dataset repo:
  6 ChromaDB collections (~4GB, within 50GB free limit)
  Downloaded by Spaces on first startup, cached permanently

Vercel:
  Next.js frontend only — no data, no models
```

---

## Project Structure

```
NyayaLex.AI/
├── src/
│   ├── ingestion/
│   │   ├── chunker.py           # shared chunker — token-aware, sentence-safe, overlapping
│   │   ├── us_code.py           # all 54 US Code titles XML → us_statutes.jsonl
│   │   ├── court_listener.py    # 28k SCOTUS opinions → us_case_law.jsonl
│   │   ├── cfr.py               # CFR selected titles → us_regulations.jsonl
│   │   ├── india_code.py        # India Code API → india_statutes.jsonl
│   │   ├── india_constitution.py# Constitution of India → india_constitution.jsonl
│   │   └── india_kanoon.py      # SC India judgments → india_case_law.jsonl
│   ├── vectorstore/             # 6 ChromaDB collections, BM25 indexes, hybrid retriever, reranker
│   ├── llm/                     # Groq client, prompt templates
│   ├── agent/
│   │   ├── graph.py             # LangGraph graph — reason→retrieve→check→generate loop
│   │   ├── nodes.py             # Node implementations
│   │   ├── state.py             # LegalResearchState TypedDict
│   │   └── tools.py             # RAG tools per collection + HyDE
│   ├── memory/                  # ConversationSummaryMemory, session store
│   ├── router/                  # Jurisdiction router + legal guardrail
│   ├── pipeline/                # NLI citation faithfulness, conflict detector
│   ├── predictor/               # legal-bert-base-uncased outcome predictor
│   ├── document/                # PDF upload parser, per-session ChromaDB
│   ├── evaluation/              # ROUGE, BERTScore, RAGAS, golden QA pairs
│   ├── api/                     # FastAPI routes + schemas
│   └── tracking/                # MLflow tracker
├── frontend/                    # Next.js app (Vercel)
│   └── src/
│       ├── app/                 # layout, page
│       └── components/          # ChatWindow, ChatInput, MessageBubble, CitationPanel, VoiceButton, OutcomeBadge
├── data/
│   ├── raw/                     # gitignored — US Code XML, SCOTUS JSON, India Code JSON
│   ├── processed/               # gitignored — 6 .jsonl files
│   ├── vectorstore/             # gitignored — 6 ChromaDB collections
│   └── eval/                    # golden_qa.jsonl (committed — hand-crafted, small)
├── scripts/
│   ├── ingest_all.py            # runs all 6 ingestion modules in sequence
│   ├── upload_to_hub.py         # uploads built ChromaDB to HuggingFace Hub Dataset
│   └── setup.sh                 # install deps, run ingestion, start Docker
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Build Phases

| Phase | What | Module |
|---|---|---|
| 1 | US Code ingestion — all 54 titles XML | `src/ingestion/us_code.py` |
| 2 | CourtListener — 28k SCOTUS (majority + dissent + concurrence) | `src/ingestion/court_listener.py` |
| 3 | CFR ingestion — Code of Federal Regulations (selected titles) | `src/ingestion/cfr.py` |
| 4 | India Code — all central acts via India Code API | `src/ingestion/india_code.py` |
| 5 | Constitution of India + Indian Kanoon SC judgments | `src/ingestion/india_constitution.py` + `india_kanoon.py` |
| 6 | Shared chunker + metadata tagger for all 6 sources | `src/ingestion/chunker.py` |
| 7 | Vector store — 6 ChromaDB collections + BM25 indexes | `src/vectorstore/` |
| 8 | LLM setup — Groq client + prompt templates | `src/llm/` |
| 9 | Jurisdiction router + legal guardrail | `src/router/` |
| 10 | LangGraph multi-hop agent — state, graph, nodes, tools | `src/agent/` |
| 11 | Hybrid RAG pipeline — HyDE + dense + BM25 + RRF + reranker | `src/vectorstore/hybrid.py` |
| 12 | NLI citation faithfulness + conflict detector | `src/pipeline/` |
| 13 | Multi-turn memory — ConversationSummaryMemory | `src/memory/` |
| 14 | Case outcome predictor — legal-bert-base-uncased | `src/predictor/` |
| 15 | Document upload + targeted RAG — PyMuPDF | `src/document/` |
| 16 | Evaluation — RAGAS + ROUGE + BERTScore + 50–100 golden QA pairs | `src/evaluation/` |
| 17 | MLflow experiment tracking — 5 runs showing improvement | `src/tracking/` |
| 18 | FastAPI backend — all routes + schemas | `src/api/` |
| 19 | Next.js frontend — chat, voice, citations, outcome, doc upload | `frontend/` |
| 20 | Docker + HuggingFace Spaces (backend) + Vercel (frontend) deploy | `docker-compose.yml` |

---

## Setup

```bash
# 1. Clone and install
git clone https://github.com/RohanPrajapati2610/NyayaLex.AI.git
cd NyayaLex.AI
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Add to .env:
#   GROQ_API_KEY=your_key_here
#   HF_TOKEN=your_huggingface_token

# 3. Run all ingestion (one time — takes ~2 hours, builds ~4GB ChromaDB)
python scripts/ingest_all.py

# 4. Upload ChromaDB to HuggingFace Hub
python scripts/upload_to_hub.py

# 5. Start backend
docker compose up
```

Frontend:

```bash
cd frontend
npm install
npm run dev
# http://localhost:3000
```

API docs: `http://localhost:8000/docs`

---

## Notes

- `data/` is fully gitignored — raw XML, processed JSONL, and ChromaDB never touch GitHub
- ChromaDB is hosted on HuggingFace Hub Dataset repo — downloaded by Spaces on first startup
- Raw XML/JSON files (~7GB) are only needed during ingestion on your laptop — delete after
- BGE embeddings require query prefix: `"Represent this sentence for searching relevant passages: "` on queries only
- CourtListener rate limit: 5,000 requests/day — ingestion uses pagination + exponential backoff
- Indian Kanoon rate limit: 1,000 requests/day — ingestion handles this automatically
- All 54 US Code titles sourced from uscode.house.gov (official US government, no license required)
