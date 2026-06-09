"""
Mode 2: Obligations.

Given an AI system description, produces a practical compliance checklist
showing which EU AI Act articles apply, what concrete actions are required,
and who is responsible (provider, deployer, or both).

If no risk_category is supplied, the system is classified first via Mode 1.
"""

import json
import re
from typing import Dict, Optional

from src.retrieval import retrieve
from src.llm import generate
from src.classify import classify_ai_system

TIER_SUMMARY = {
    "PROHIBITED": (
        "the system is banned under Article 5 and cannot be lawfully deployed. "
        "No compliance pathway exists - the activity itself must cease."
    ),
    "HIGH": (
        "extensive compliance obligations apply, including a risk management system "
        "(Article 9), data governance (Article 10), technical documentation (Article 11), "
        "record-keeping (Article 12), transparency to deployers (Article 13), human "
        "oversight (Article 14), accuracy and cybersecurity (Article 15), quality "
        "management system (Article 17), conformity assessment (Article 43), CE marking "
        "(Article 48), EU database registration (Article 49), post-market monitoring "
        "(Article 72), and serious incident reporting (Article 73)."
    ),
    "LIMITED": (
        "transparency obligations under Article 50 apply: inform users they are "
        "interacting with AI, label AI-generated synthetic content, disclose deepfakes "
        "and AI-generated text on matters of public interest."
    ),
    "MINIMAL": (
        "no specific obligations apply under the EU AI Act. Voluntary codes of conduct "
        "(Article 95) are encouraged but not required."
    ),
}

OBLIGATIONS_QUERY_TERMS = {
    "HIGH": (
        "high-risk AI system obligations provider deployer requirements risk management "
        "data governance technical documentation human oversight conformity assessment "
        "CE marking registration post-market monitoring serious incidents"
    ),
    "LIMITED": (
        "transparency obligations chatbot AI generated content deepfake Article 50 "
        "disclosure inform users"
    ),
    "PROHIBITED": "prohibited AI practices Article 5 banned use cases",
    "MINIMAL": "voluntary codes of conduct minimal risk Article 95",
}

OBLIGATIONS_PROMPT = """You are an expert on EU AI Act compliance obligations.

The user's AI system has been classified as {risk_category} risk. Your task is to
produce a practical compliance checklist showing the specific obligations that apply
to this exact system, with article citations and concrete required actions.

For {risk_category} risk systems: {tier_summary}

AI SYSTEM:
{system_description}

RELEVANT EU AI ACT EXCERPTS:
{context}

Respond ONLY with valid JSON in exactly this format:
{{
  "risk_category": "{risk_category}",
  "applicable_role": "PROVIDER" or "DEPLOYER" or "BOTH",
  "obligations": [
    {{
      "title": "Short title (e.g., 'Risk Management System')",
      "article": "Article reference (e.g., 'Article 9')",
      "description": "1-2 sentence description of what is required for this specific system",
      "required_actions": ["Concrete action 1", "Concrete action 2"],
      "priority": "MANDATORY" or "RECOMMENDED"
    }}
  ],
  "summary": "1-2 sentence summary of the total compliance burden for this system"
}}

Guidance per tier:
- PROHIBITED: Return a single obligation explaining the system cannot be deployed and what alternative actions exist.
- HIGH: List 8-12 obligations covering Articles 9-15, 17, 43, 48, 49, 72, 73 as applicable.
- LIMITED: Focus on Article 50 transparency duties. 2-3 obligations.
- MINIMAL: One obligation about voluntary codes of conduct.

Make required_actions specific to the actual system described, not generic boilerplate.
"""


def get_obligations(
    system_description: str,
    risk_category: Optional[str] = None,
    k: int = 8,
) -> Dict:
    """
    Get EU AI Act obligations for an AI system.

    Args:
        system_description: Plain-language description of the AI system.
        risk_category: Optional pre-classified risk tier. If None, Mode 1 is run first.
        k: Number of EU AI Act chunks to retrieve as context.

    Returns:
        Dict with keys: risk_category, applicable_role, obligations, summary, sources.
    """
    if risk_category is None:
        classification = classify_ai_system(system_description)
        if "error" in classification:
            return {
                "error": "Classification step failed before obligations could be computed",
                "details": classification,
            }
        risk_category = classification["risk_category"]

    risk_category = risk_category.upper()
    if risk_category not in TIER_SUMMARY:
        return {"error": f"Unknown risk_category: {risk_category}"}

    retrieval_query = (
        OBLIGATIONS_QUERY_TERMS.get(risk_category, "AI Act obligations")
        + " "
        + system_description
    )
    hits = retrieve(query=retrieval_query, k=k, source_filter="eu_ai_act")

    context = "\n\n".join([
        f"[{h['source']} page {h['page']}]\n{h['text']}"
        for h in hits
    ])

    prompt = OBLIGATIONS_PROMPT.format(
        risk_category=risk_category,
        tier_summary=TIER_SUMMARY[risk_category],
        system_description=system_description,
        context=context,
    )

    response_text = generate(prompt)

    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return {"error": "No JSON in LLM response", "raw_response": response_text, "sources": hits}

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "raw_response": response_text, "sources": hits}

    result["system_description"] = system_description
    result["sources"] = hits
    return result
