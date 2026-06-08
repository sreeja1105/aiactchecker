"""
Ingest all PDFs in docs/ into ChromaDB.

Run from project root:
    python scripts/ingest_docs.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion import ingest_pdf

DOCS_DIR = PROJECT_ROOT / "docs"


def main():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {DOCS_DIR}. Run scripts/download_docs.py first.")
        sys.exit(1)

    print(f"Ingesting {len(pdfs)} document(s) into ChromaDB")
    print("-" * 78)
    for pdf in pdfs:
        ingest_pdf(pdf)
        print()
    print("Done.")


if __name__ == "__main__":
    main()
