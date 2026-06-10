"""
Test the LangGraph orchestration.

Runs four queries with different intents and verifies:
  1. The router picks the correct mode for each
  2. Mode 1 (CLASSIFY) actually returns a real classification
  3. Modes 2 to 4 return their stub response

Total LLM calls: 1 (only the CLASSIFY query hits the LLM).

Run from project root:
    python scripts/test_graph.py
"""

import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import run

SEPARATOR = "-" * 78

TEST_QUERIES = [
    {
        "label": "CLASSIFY (wired)",
        "query": (
            "Classify this system: An AI tool that screens job applicants by scoring "
            "CVs and ranking candidates for HR teams."
        ),
        "expected_mode": "CLASSIFY",
    },
    {
        "label": "OBLIGATIONS (stub)",
        "query": "What are the obligations for a high risk AI system?",
        "expected_mode": "OBLIGATIONS",
    },
    {
        "label": "GAP_ANALYSIS (stub)",
        "query": "Do a gap analysis on our current implementation state",
        "expected_mode": "GAP_ANALYSIS",
    },
    {
        "label": "CROSS_MAP (stub)",
        "query": "How does data governance compare across the EU AI Act and GDPR?",
        "expected_mode": "CROSS_MAP",
    },
]


def main():
    print("AIActChecker - LangGraph orchestration test")
    print(SEPARATOR)
    print("Mode 1 (CLASSIFY) is wired. Modes 2 to 4 are stubs.")
    print(SEPARATOR)

    pass_count = 0
    for i, case in enumerate(TEST_QUERIES, start=1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] {case['label']}")
        for line in textwrap.wrap("Query: " + case["query"], width=74):
            print(f"  {line}")

        result = run(case["query"])

        actual_mode = result.get("mode", "?")
        expected_mode = case["expected_mode"]
        routing_ok = actual_mode == expected_mode

        print(f"\n  Routing:")
        print(f"    Expected mode: {expected_mode}")
        print(f"    Actual mode:   {actual_mode}   {'[PASS]' if routing_ok else '[FAIL]'}")

        print(f"\n  Output:")
        if actual_mode == "CLASSIFY":
            cls = result.get("classification", {})
            if "error" in cls:
                print(f"    ERROR: {cls['error']}")
            else:
                print(f"    Risk category: {cls.get('risk_category', '?')}")
                refs = cls.get('article_references', [])
                if refs:
                    print(f"    Citations:     {', '.join(refs[:3])}")
                reasoning = cls.get('reasoning', '')
                if reasoning:
                    for line in textwrap.wrap(f"Reasoning: {reasoning}", width=70):
                        print(f"    {line}")
        elif actual_mode == "OBLIGATIONS":
            stub = result.get("obligations", {})
            print(f"    {stub.get('note', '?')}")
        elif actual_mode == "GAP_ANALYSIS":
            stub = result.get("gap_analysis", {})
            print(f"    {stub.get('note', '?')}")
        elif actual_mode == "CROSS_MAP":
            stub = result.get("cross_map", {})
            print(f"    {stub.get('note', '?')}")
        else:
            print(f"    Unknown mode, full state: {result}")

        if routing_ok:
            pass_count += 1

        print(SEPARATOR)

    print(f"\nRouting result: {pass_count} of {len(TEST_QUERIES)} correct\n")


if __name__ == "__main__":
    main()
