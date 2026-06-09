"""
Test the Mode 1 Classify pipeline against a range of example AI systems.

Run from project root:
    python scripts/test_classify.py
"""

import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classify import classify_ai_system

EXAMPLES = [
    # Expected: HIGH (Annex III - employment)
    "An AI tool that screens job applicants by scoring CVs and ranking candidates for HR teams",

    # Expected: MINIMAL (no specific obligations)
    "An AI-powered email spam filter for a personal mailbox",

    # Expected: PROHIBITED (Article 5 - social scoring)
    "A government-run AI system that assigns social credit scores to citizens based on their public behaviour and limits their access to public services accordingly",

    # Expected: LIMITED (Article 50 - chatbot transparency)
    "A customer service chatbot on an e-commerce website that handles refund requests and product questions",

    # Expected: PROHIBITED (Article 5 - real-time biometric ID in public)
    "A facial recognition system deployed by law enforcement in real time across public spaces to identify individuals",

    # Expected: HIGH (Annex III - education)
    "An AI system that evaluates students' exam answers and assigns grades in a university course",
]

SEPARATOR = "-" * 78


def main():
    print("AIActChecker - Mode 1 (Classify) test")
    print(SEPARATOR)

    for i, description in enumerate(EXAMPLES, start=1):
        print(f"\n[{i}/{len(EXAMPLES)}] System:")
        for line in textwrap.wrap(description, width=72):
            print(f"  {line}")

        result = classify_ai_system(description)

        if "error" in result:
            print(f"\n  ERROR: {result['error']}")
            print(f"  Raw response: {result.get('raw_response', 'N/A')[:200]}")
        else:
            print(f"\n  Risk category: {result['risk_category']}")
            print(f"  Confidence:    {result['confidence']}")
            print(f"  Articles:      {', '.join(result['relevant_articles'])}")
            print(f"  Reasoning:")
            for line in textwrap.wrap(result['reasoning'], width=72):
                print(f"    {line}")
            print(f"  Top source:    {result['sources'][0]['source']} page {result['sources'][0]['page']}  (sim {result['sources'][0]['score']})")

        print(f"\n{SEPARATOR}")


if __name__ == "__main__":
    main()
