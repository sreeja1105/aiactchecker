"""
Test Mode 3 Gap Analysis with pre-classified risk categories (saves LLM calls).

Run from project root:
    python scripts/test_gap_analysis.py
"""

import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.gap_analysis import analyze_gaps

# Pre-classify systems to skip the classification LLM call during testing.
# In production the API would call classify_ai_system() first.
EXAMPLES = [
    {
        "system": (
            "An AI tool that screens job applicants by scoring CVs and ranking "
            "candidates for HR teams"
        ),
        "risk_category": "HIGH",
        "current_state": (
            "We use the AI scores as one of several inputs to HR decisions, and a "
            "human always reviews the top 20 candidates before any interview is "
            "scheduled. We log all model outputs in our HR system for audit purposes. "
            "We have not yet documented our risk management process or our training "
            "data sources. No formal post-market monitoring is in place. We have "
            "basic security on the AI service (SSO, encrypted storage). No CE marking "
            "process started, no EU database registration."
        ),
    },
]

SEPARATOR = "-" * 78


def main():
    print("AIActChecker - Mode 3 (Gap Analysis) test")
    print(SEPARATOR)

    for i, case in enumerate(EXAMPLES, start=1):
        print(f"\n[{i}/{len(EXAMPLES)}] System:")
        for line in textwrap.wrap(case["system"], width=72):
            print(f"  {line}")
        print(f"\n  Pre-classified as: {case['risk_category']}")
        print(f"\n  Current state described:")
        for line in textwrap.wrap(case["current_state"], width=72):
            print(f"    {line}")

        result = analyze_gaps(
            case["system"],
            case["current_state"],
            risk_category=case["risk_category"],
        )

        if "error" in result:
            print(f"\n  ERROR: {result['error']}")
            print(f"  Raw: {result.get('raw_response', 'N/A')[:300]}")
            continue

        s = result.get("summary", {})
        print(f"\n  Risk category:    {result['risk_category']}")
        print(f"  Compliance score: {s.get('compliance_score_percent', '?')}%")
        print(f"  Met / Partial / Gap:  {s.get('met', '?')} / {s.get('partial', '?')} / {s.get('gaps', '?')}")
        print(f"  Critical gaps:    {s.get('critical_gaps_count', '?')}")

        print(f"\n  Detailed gap analysis:")
        for item in result.get("gap_analysis", []):
            mark = {"MET": "[+]", "PARTIAL": "[~]", "GAP": "[-]"}.get(item["status"], "[?]")
            print(f"\n    {mark} {item['requirement_title']}  [{item['article']}]")
            print(f"        Status: {item['status']}  |  Priority: {item['priority']}")
            print(f"        Currently: {item['current_state_assessment']}")
            if item["status"] != "MET":
                if item.get("gap_description"):
                    print(f"        Missing:   {item['gap_description']}")
                if item.get("recommended_action"):
                    print(f"        Action:    {item['recommended_action']}")

        print(f"\n{SEPARATOR}")


if __name__ == "__main__":
    main()
