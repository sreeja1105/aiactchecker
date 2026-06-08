"""
Sanity-check the RAG pipeline with sample queries about the EU AI Act.

Run from project root:
    python scripts/test_query.py
"""

import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import answer_with_citations

TEST_QUERIES = [
    "What is a high-risk AI system under the EU AI Act?",
    "What are the obligations for providers of high-risk AI systems?",
    "What is the scope of the EU AI Act?",
    "When does the EU AI Act enter into force?",
]

SEPARATOR = "-" * 78


def main():
    print("AIActChecker - retrieval test")
    print(SEPARATOR)

    for i, query in enumerate(TEST_QUERIES, start=1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] Query: {query}\n")
        result = answer_with_citations(query, k=4)

        print("Answer:")
        for line in textwrap.wrap(result["answer"], width=78):
            print(f"  {line}")

        print("\nRetrieved sources:")
        for j, src in enumerate(result["sources"], start=1):
            print(f"  {j}. {src['source']} page {src['page']}  (similarity {src['score']})")

        print(f"\n{SEPARATOR}")


if __name__ == "__main__":
    main()
