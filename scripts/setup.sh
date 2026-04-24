#!/usr/bin/env bash
set -e

echo "=================================================="
echo "  NyayaLex.AI — Setup"
echo "=================================================="

# 1. Install Python dependencies
echo ""
echo "==> Installing Python dependencies..."
pip install -r requirements.txt

# 2. Check required env vars
if [ -z "$GROQ_API_KEY" ]; then
  echo "ERROR: GROQ_API_KEY is not set. Copy .env.example to .env and fill it in."
  exit 1
fi
echo "==> GROQ_API_KEY found."

if [ -z "$HF_TOKEN" ]; then
  echo "WARN: HF_TOKEN not set. You will need it to upload ChromaDB to HuggingFace Hub."
fi

# 3. Create data directories (gitignored — never committed)
echo ""
echo "==> Creating data directories..."
mkdir -p data/raw/us_code
mkdir -p data/raw/court_listener
mkdir -p data/raw/cfr
mkdir -p data/raw/india_code
mkdir -p data/raw/india_constitution
mkdir -p data/raw/india_kanoon
mkdir -p data/processed
mkdir -p data/vectorstore
mkdir -p data/eval

# 4. Run all ingestion phases
echo ""
echo "==> Running ingestion (all 6 phases — takes ~2 hours)..."
echo "    Phase 1: US Code — all 54 titles"
echo "    Phase 2: CourtListener — 28k SCOTUS opinions"
echo "    Phase 3: CFR — Code of Federal Regulations"
echo "    Phase 4: India Code — all central acts"
echo "    Phase 5: Constitution of India + SC India judgments"
echo ""
python scripts/ingest_all.py

# 5. Start services
echo ""
echo "==> Starting services with Docker Compose..."
docker compose up -d

echo ""
echo "=================================================="
echo "  Setup complete."
echo ""
echo "  API docs:  http://localhost:8000/docs"
echo "  MLflow:    http://localhost:5000"
echo "  ChromaDB:  http://localhost:8001"
echo ""
echo "  Frontend (run separately):"
echo "    cd frontend && npm install && npm run dev"
echo "    http://localhost:3000"
echo ""
echo "  To upload ChromaDB to HuggingFace Hub:"
echo "    python scripts/upload_to_hub.py"
echo "=================================================="
