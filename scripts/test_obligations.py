"""
Test the Mode 2 Obligations pipeline.

Run from project root:
    python scripts/test_obligations.py
"""

import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.obligations import get_obligations

EXAMPLES = [
    "An AI tool that screens job applicants by scoring CVs and ranking candidates for HR teams",
]
SEPARATOR = "-" * 78


def main():
    print("AIActChecker - Mode 2 (Obligations) test")
    print(SEPARATOR)

    for i, description in enumerate(EXAMPLES, start=1):
        print(f"\n[{i}/{len(EXAMPLES)}] System:")
        for line in textwrap.wrap(description, width=72):
            print(f"  {line}")

        result = get_obligations(description)

        if "error" in result:
            print(f"\n  ERROR: {result['error']}")
            continue

        print(f"\n  Risk category:   {result['risk_category']}")
        print(f"  Applicable role: {result['applicable_role']}")
        print(f"  Obligations:     {len(result['obligations'])}")
        print(f"  Summary:")
        for line in textwrap.wrap(result['summary'], width=72):
            print(f"    {line}")

        print(f"\n  Compliance checklist:")
        for j, ob in enumerate(result['obligations'], start=1):
            print(f"    {j}. {ob['title']}  [{ob['article']}]  ({ob['priority']})")
            for line in textwrap.wrap(ob['description'], width=66):
                print(f"       {line}")
            for action in ob['required_actions']:
                print(f"       - {action}")

        print(f"\n{SEPARATOR}")


if __name__ == "__main__":
    main()
