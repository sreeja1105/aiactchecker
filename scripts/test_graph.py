"""
Test the LangGraph orchestration with ALL 4 modes wired.

Runs queries that route to each of the 4 modes. Each test produces real output
(not stub responses).

Total LLM calls: ~5 to 7 depending on Mode 3 needing classification.
The 4 calls hit the per minute rate limit so retry logic will pause between
them. Full run takes ~3 to 5 minutes.

Run from project root:
    python scripts/test_graph.py
"""

import sys
import textwrap
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import run

SEPARATOR = "-" * 78


def print_classification(cls):
    if "error" in cls:
        print(f"    ERROR: {cls['error']}")
        return
    print(f"    Risk category: {cls.get('risk_category', '?')}")
    refs = cls.get("article_references", [])
    if refs:
        print(f"    Citations:     {', '.join(refs[:3])}")
    reasoning = cls.get("reasoning", "")
    if reasoning:
        for line in textwrap.wrap(f"Reasoning: {reasoning}", width=70):
            print(f"    {line}")


def print_obligations(ob):
    if "error" in ob:
        print(f"    ERROR: {ob['error']}")
        return
    risk = ob.get("risk_category", "?")
    print(f"    Risk category: {risk}")
    obligations_list = ob.get("obligations", [])
    print(f"    Obligations returned: {len(obligations_list)}")
    for i, item in enumerate(obligations_list[:3], start=1):
        title = item.get("title", "?")
        article = item.get("article", "?")
        print(f"      {i}. [{article}] {title}")
    if len(obligations_list) > 3:
        print(f"      ... and {len(obligations_list) - 3} more")


def print_gap_analysis(ga):
    if "error" in ga:
        print(f"    ERROR: {ga['error']}")
        return
    s = ga.get("summary", {})
    print(f"    Risk category:    {ga.get('risk_category', '?')}")
    print(f"    Compliance score: {s.get('compliance_score_percent', '?')}%")
    print(f"    Met / Partial / Gap: {s.get('met', '?')} / {s.get('partial', '?')} / {s.get('gaps', '?')}")
    print(f"    Critical gaps:    {s.get('critical_gaps_count', '?')}")


def print_cross_map(cm):
    if "error" in cm:
        print(f"    ERROR: {cm['error']}")
        return
    print(f"    Topic: {cm.get('topic', '?')}")
    mappings = cm.get("mappings", {})
    for key, label in [("eu_ai_act", "EU AI Act"), ("gdpr", "GDPR"), ("nist_ai_rmf", "NIST AI RMF")]:
        block = mappings.get(key, {})
        refs = block.get("primary_references", [])
        print(f"    {label}: {', '.join(refs[:3]) if refs else '(none)'}")


def main():
    print("AIActChecker - LangGraph orchestration test (all 4 modes wired)")
    print(SEPARATOR)

    # ----- Test 1: CLASSIFY -----
    print("\n[1/4] CLASSIFY")
    query1 = "Classify this system: An AI tool that screens job applicants by scoring CVs"
    print(f"  Query: {query1}")
    result = run(query1)
    print(f"  Mode routed: {result.get('mode')}")
    print("  Output:")
    print_classification(result.get("classification", {}))
    print(SEPARATOR)

    # ----- Test 2: OBLIGATIONS -----
    print("\n[2/4] OBLIGATIONS")
    query2 = "What obligations apply to an AI tool that screens job applicants?"
    print(f"  Query: {query2}")
    print("  Passing pre classified risk_category=HIGH to save an LLM call.")
    result = run(query2, risk_category="HIGH")
    print(f"  Mode routed: {result.get('mode')}")
    print("  Output:")
    print_obligations(result.get("obligations", {}))
    print(SEPARATOR)

    # ----- Test 3: GAP_ANALYSIS -----
    print("\n[3/4] GAP_ANALYSIS")
    query3 = "Do a gap analysis on our current implementation"
    system3 = "An AI tool that screens job applicants by scoring CVs and ranking candidates for HR teams"
    current3 = (
        "We use the AI scores as one of several inputs to HR decisions. A human reviews "
        "the top 20 candidates before any interview is scheduled. We log all model "
        "outputs for audit. We have not documented our risk management process or "
        "training data sources. No formal post market monitoring is in place. Basic "
        "security on the AI service (SSO, encrypted storage). No CE marking, no EU "
        "database registration."
    )
    print(f"  Query: {query3}")
    print("  Passing system_description, current_state, and risk_category=HIGH.")
    result = run(
        query3,
        system_description=system3,
        current_state=current3,
        risk_category="HIGH",
    )
    print(f"  Mode routed: {result.get('mode')}")
    print("  Output:")
    print_gap_analysis(result.get("gap_analysis", {}))
    print(SEPARATOR)

    # ----- Test 4: CROSS_MAP -----
    print("\n[4/4] CROSS_MAP")
    query4 = "Compare data governance across EU AI Act and GDPR"
    print(f"  Query: {query4}")
    result = run(query4, topic="Data Governance and Data Quality for AI Systems")
    print(f"  Mode routed: {result.get('mode')}")
    print("  Output:")
    print_cross_map(result.get("cross_map", {}))
    print(SEPARATOR)

    print("\nAll 4 modes routed and executed end to end via LangGraph.\n")


if __name__ == "__main__":
    main()
