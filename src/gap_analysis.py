"""
Mode 3: Gap Analysis.

Compares an AI system's current implementation state against EU AI Act
obligations for its risk tier. Identifies which obligations are MET, PARTIAL,
or GAP, with priority and recommended actions.

Efficient single-LLM-call design:
  - Obligations are hardcoded per risk tier (they come from the law, not the LLM)
  - Optional pre-classification skips the Mode 1 call if risk is known
  - Result: 1 LLM call per gap analysis (or 2 if classification needed)
"""

import json
import re
from typing import Dict, Optional

from src.llm import generate
from src.classify import classify_ai_system

# Hardcoded EU AI Act obligations per risk tier
OBLIGATIONS_BY_TIER = {
    "HIGH": [
        {"title": "Risk Management System", "article": "Article 9"},
        {"title": "Data Governance", "article": "Article 10"},
        {"title": "Technical Documentation", "article": "Article 11"},
        {"title": "Record-Keeping", "article": "Article 12"},
        {"title": "Transparency to Deployers", "article": "Article 13"},
        {"title": "Human Oversight", "article": "Article 14"},
        {"title": "Accuracy, Robustness, Cybersecurity", "article": "Article 15"},
        {"title": "Quality Management System", "article": "Article 17"},
        {"title": "Conformity Assessment", "article": "Article 43"},
        {"title": "CE Marking", "article": "Article 48"},
        {"title": "EU Database Registration", "article": "Article 49"},
        {"title": "Post-Market Monitoring", "article": "Article 72"},
        {"title": "Serious Incident Reporting", "article": "Article 73"},
    ],
    "LIMITED": [
        {"title": "AI Interaction Disclosure", "article": "Article 50(1)"},
        {"title": "AI-Generated Content Marking", "article": "Article 50(2)"},
        {"title": "Deepfake Disclosure", "article": "Article 50(4)"},
    ],
    "MINIMAL": [
        {"title": "Voluntary Codes of Conduct", "article": "Article 95"},
    ],
}

GAP_PROMPT = """You are an EU AI Act compliance auditor.

An AI system has been classified as {risk_category} risk. Produce a gap analysis
between the user's current implementation and the EU AI Act requirements.

REQUIREMENTS THAT APPLY ({total} total):
{requirements_list}

AI SYSTEM: {system_description}

CURRENT IMPLEMENTATION STATE:
{current_state}

For each requirement, assess whether the user's current state addresses it:
- MET: fully satisfies
- PARTIAL: partly addresses with clear gaps remaining
- GAP: not addressed at all

Respond ONLY with valid JSON:
{{
  "gap_analysis": [
    {{
      "requirement_title": "...",
      "article": "...",
      "status": "MET" or "PARTIAL" or "GAP",
      "current_state_assessment": "What the user has that relates to this requirement",
      "gap_description": "What is missing or weak (empty string if MET)",
      "priority": "CRITICAL" or "HIGH" or "MEDIUM" or "LOW",
      "recommended_action": "Specific concrete next step (empty if MET)"
    }}
  ],
  "summary": {{
    "total_requirements": <number>,
    "met": <number>,
    "partial": <number>,
    "gaps": <number>,
    "compliance_score_percent": <0 to 100>,
    "critical_gaps_count": <number>
  }}
}}

Priority guidance:
- CRITICAL: GAP on safety/rights obligations (Articles 9, 14, 15)
- HIGH: GAP on documentation/transparency (Articles 10, 11, 13)
- MEDIUM: GAP on process obligations (Articles 17, 49, 72)
- LOW: PARTIAL items where most work is done
"""


def analyze_gaps(
    system_description: str,
    current_state: str,
    risk_category: Optional[str] = None,
) -> Dict:
    """
    Gap analysis with a single LLM call.

    Args:
        system_description: Plain-language description of the AI system.
        current_state: Description of compliance controls the user currently has.
        risk_category: Pre-classified risk tier (PROHIBITED/HIGH/LIMITED/MINIMAL).
            If None, classification runs first (adds one LLM call).
    """
    if risk_category is None:
        classification = classify_ai_system(system_description)
        if "error" in classification:
            return {"error": "Classification failed", "details": classification}
        risk_category = classification["risk_category"]

    risk_category = risk_category.upper()

    if risk_category == "PROHIBITED":
        return {
            "system_description": system_description,
            "current_state_input": current_state,
            "risk_category": "PROHIBITED",
            "gap_analysis": [{
                "requirement_title": "System cannot be deployed",
                "article": "Article 5",
                "status": "GAP",
                "current_state_assessment": "The system falls under prohibited AI practices.",
                "gap_description": "System violates Article 5 and cannot be brought into compliance.",
                "priority": "CRITICAL",
                "recommended_action": "Halt all deployment. Redesign to avoid prohibited use cases.",
            }],
            "summary": {
                "total_requirements": 1, "met": 0, "partial": 0,
                "gaps": 1, "compliance_score_percent": 0, "critical_gaps_count": 1,
            },
        }

    obligations = OBLIGATIONS_BY_TIER.get(risk_category, [])
    if not obligations:
        return {"error": f"No obligations defined for risk category: {risk_category}"}

    requirements_list = "\n".join([
        f"- {ob['title']} ({ob['article']})" for ob in obligations
    ])

    prompt = GAP_PROMPT.format(
        risk_category=risk_category,
        total=len(obligations),
        requirements_list=requirements_list,
        system_description=system_description,
        current_state=current_state,
    )

    response_text = generate(prompt)

    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return {"error": "No JSON in LLM response", "raw_response": response_text}

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "raw_response": response_text}

    result["system_description"] = system_description
    result["current_state_input"] = current_state
    result["risk_category"] = risk_category

    return result
