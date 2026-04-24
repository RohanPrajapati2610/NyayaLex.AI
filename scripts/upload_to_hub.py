"""
Upload pre-built ChromaDB vector store to HuggingFace Hub Dataset repo.

Run this AFTER scripts/ingest_all.py + Phase 7 (vector store build) are complete.
The Dataset repo is separate from the Spaces repo — it stores large binary files
using git-lfs (up to 50GB on the free tier).

Usage:
    python scripts/upload_to_hub.py

Requirements:
    HF_TOKEN and HF_DATASET_REPO in .env
    huggingface_hub installed (already in requirements.txt)

What this does:
    1. Creates the Dataset repo on HuggingFace Hub if it doesn't exist
    2. Uploads data/vectorstore/ (all 6 ChromaDB collections, ~4GB) to the repo
    3. On HuggingFace Spaces startup, the app downloads this repo automatically
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, snapshot_download

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "RohanPrajapati2610/nyayalex-vectorstore")
LOCAL_VECTORSTORE = Path("data/vectorstore")


def upload() -> None:
    if not HF_TOKEN:
        print("ERROR: HF_TOKEN not set in .env")
        sys.exit(1)

    if not LOCAL_VECTORSTORE.exists() or not any(LOCAL_VECTORSTORE.iterdir()):
        print(f"ERROR: {LOCAL_VECTORSTORE} is empty. Run Phase 7 (vector store build) first.")
        sys.exit(1)

    api = HfApi(token=HF_TOKEN)

    # Create the Dataset repo if it doesn't exist
    try:
        create_repo(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN,
            exist_ok=True,
            private=True,  # keep private — only your Spaces app accesses it
        )
        print(f"Dataset repo ready: {HF_DATASET_REPO}")
    except Exception as e:
        print(f"ERROR creating repo: {e}")
        sys.exit(1)

    # Upload the entire vectorstore directory
    print(f"Uploading {LOCAL_VECTORSTORE} → {HF_DATASET_REPO} ...")
    print("This may take 20-40 minutes for ~4GB. Do not interrupt.")

    api.upload_folder(
        folder_path=str(LOCAL_VECTORSTORE),
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        path_in_repo="vectorstore",
        token=HF_TOKEN,
        commit_message="Upload ChromaDB vectorstore — all 6 collections",
    )

    print(f"\nUpload complete.")
    print(f"Dataset repo: https://huggingface.co/datasets/{HF_DATASET_REPO}")
    print(f"Your HuggingFace Spaces app will download this on first startup.")


if __name__ == "__main__":
    upload()
