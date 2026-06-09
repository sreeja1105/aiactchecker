"""
Test Mode 4 Cross-Standard Map.

Run from project root:
    python scripts/test_cross_map.py
"""

import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cross_map import cross_map

# Single topic for first test run (saves API budget)
TOPICS = [
    "Data Governance and Data Quality for AI Systems",
]

SEPARATOR = "-" * 78


def main():
    print("AIActChecker - Mode 4 (Cross-Standard Map) test")
    print(SEPARATOR)

    for i, topic in enumerate(TOPICS, start=1):
        print(f"\n[{i}/{len(TOPICS)}] Topic: {topic}")

        result = cross_map(topic)

        if "error" in result:
            print(f"\n  ERROR: {result['error']}")
            print(f"  Raw: {result.get('raw_response', 'N/A')[:300]}")
            continue

        for framework_key, framework_label in [
            ("eu_ai_act", "EU AI Act"),
            ("gdpr", "GDPR"),
            ("nist_ai_rmf", "NIST AI RMF"),
        ]:
            block = result["mappings"][framework_key]
            print(f"\n  >>> {framework_label}")
            refs = ", ".join(block.get("primary_references", []))
            print(f"      References: {refs if refs else '(none)'}")
            print(f"      Summary:")
            for line in textwrap.wrap(block["summary"], width=70):
                print(f"        {line}")
            if block.get("key_obligations"):
                print(f"      Key obligations:")
                for ob in block["key_obligations"]:
                    print(f"        - {ob}")

        print(f"\n  >>> Overlap")
        for line in textwrap.wrap(result.get("overlap", ""), width=72):
            print(f"      {line}")

        print(f"\n  >>> Differences")
        for line in textwrap.wrap(result.get("differences", ""), width=72):
            print(f"      {line}")

        print(f"\n  >>> Compliance guidance")
        for line in textwrap.wrap(result.get("compliance_guidance", ""), width=72):
            print(f"      {line}")

        print(f"\n{SEPARATOR}")


if __name__ == "__main__":
    main()
