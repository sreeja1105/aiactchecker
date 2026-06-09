"""
Sanity-check the RAG pipeline with sample queries.

EU-AI-Act-specific queries are filtered to source=eu_ai_act to avoid
cross-corpus pollution. The final cross-framework query searches all sources.

Run from project root:
    python scripts/test_query.py
"""

import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval import answer_with_citations

# (query, source_filter) tuples
TEST_QUERIES = [
    ("What is a high-risk AI system under the EU AI Act?", "eu_ai_act"),
    ("What are the obligations for providers of high-risk AI systems?", "eu_ai_act"),
    ("What is the scope of the EU AI Act?", "eu_ai_act"),
    ("When does the EU AI Act enter into force?", "eu_ai_act"),
    ("How does the EU AI Act relate to GDPR for biometric data?", None),  # cross-framework
]

SEPARATOR = "-" * 78


def main():
    print("AIActChecker - retrieval test")
    print(SEPARATOR)

    for i, (query, source) in enumerate(TEST_QUERIES, start=1):
        filter_note = f"(filtered to {source})" if source else "(all sources)"
        print(f"\n[{i}/{len(TEST_QUERIES)}] {filter_note}")
        print(f"Query: {query}\n")

        result = answer_with_citations(query, k=4, source_filter=source)

        print("Answer:")
        for line in textwrap.wrap(result["answer"], width=78):
            print(f"  {line}")

        print("\nRetrieved sources:")
        for j, src in enumerate(result["sources"], start=1):
            print(f"  {j}. {src['source']} page {src['page']}  (similarity {src['score']})")

        print(f"\n{SEPARATOR}")


if __name__ == "__main__":
    main()
