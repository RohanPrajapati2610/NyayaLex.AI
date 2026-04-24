"""
Master ingestion script — runs all 6 data ingestion phases in sequence.

Run once on your laptop to build all processed JSONL files.
After this completes, run scripts/upload_to_hub.py to push ChromaDB to HuggingFace.

Usage:
    python scripts/ingest_all.py                  # run all phases
    python scripts/ingest_all.py --phases 1 2 3   # run specific phases only

Phases:
    1 — US Code (all 54 titles)           → data/processed/us_statutes.jsonl
    2 — CourtListener SCOTUS (28k cases)  → data/processed/us_case_law.jsonl
    3 — CFR selected titles               → data/processed/us_regulations.jsonl
    4 — India Code (all central acts)     → data/processed/india_statutes.jsonl
    5 — Constitution of India             → data/processed/india_constitution.jsonl
    6 — Indian Kanoon (SC India, 5k)      → data/processed/india_case_law.jsonl
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on path when run as script
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_phase(phase_num: int) -> None:
    start = time.time()
    print(f"\n{'='*60}")
    print(f"  PHASE {phase_num}")
    print(f"{'='*60}")

    if phase_num == 1:
        from src.ingestion.us_code import ingest
        count = ingest()
        print(f"Phase 1 complete — {count:,} chunks in {time.time()-start:.0f}s")

    elif phase_num == 2:
        from src.ingestion.court_listener import ingest
        count = ingest()
        print(f"Phase 2 complete — {count:,} chunks in {time.time()-start:.0f}s")

    elif phase_num == 3:
        from src.ingestion.cfr import ingest
        count = ingest()
        print(f"Phase 3 complete — {count:,} chunks in {time.time()-start:.0f}s")

    elif phase_num == 4:
        from src.ingestion.india_code import ingest
        count = ingest()
        print(f"Phase 4 complete — {count:,} chunks in {time.time()-start:.0f}s")

    elif phase_num == 5:
        from src.ingestion.india_constitution import ingest as ingest_constitution
        from src.ingestion.india_kanoon import ingest as ingest_kanoon
        c1 = ingest_constitution()
        c2 = ingest_kanoon()
        print(f"Phase 5 complete — {c1+c2:,} chunks in {time.time()-start:.0f}s")

    else:
        print(f"Unknown phase: {phase_num}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NyayaLex.AI — run all ingestion phases")
    parser.add_argument(
        "--phases",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        help="Which phases to run (default: all)",
    )
    args = parser.parse_args()

    # Ensure directories exist
    for d in ["data/raw", "data/processed", "data/vectorstore", "data/eval"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    for phase in sorted(args.phases):
        run_phase(phase)

    elapsed = time.time() - total_start
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"\n{'='*60}")
    print(f"  All phases complete in {mins}m {secs}s")
    print(f"  Processed files in: data/processed/")
    print(f"  Next step: python scripts/upload_to_hub.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
