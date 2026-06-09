"""
Mode 1: Classify an AI system into EU AI Act risk categories.

Risk tiers per EU AI Act:
  PROHIBITED  (Article 5)       Banned outright. Social scoring, manipulation, etc.
  HIGH        (Article 6 + Annex III)  Strict obligations. Hiring, education, etc.
  LIMITED     (Article 50)      Transparency obligations only. Chatbots, deepfakes.
  MINIMAL                       No specific obligations. Spam filters, game AI, etc.
"""

import json
import re
from typing import Dict

from src.retrieval import retrieve
from src.llm import generate

VALID_CATEGORIES = {"PROHIBITED", "HIGH", "LIMITED", "MINIMAL"}

CLASSIFICATION_PROMPT = """You are an expert on the EU AI Act risk classification framework.

The EU AI Act defines four risk categories:
- PROHIBITED (Article 5): Banned AI practices including social scoring by public authorities, real-time remote biometric identification in publicly accessible spaces for law enforcement (with narrow exceptions), AI for manipulation causing harm, exploitation of vulnerabilities, untargeted scraping of facial images, emotion recognition in workplace/education, biometric categorization to infer sensitive attributes.
- HIGH (Article 6 + Annex III): AI systems used as safety components of regulated products, OR systems listed in Annex III: biometric identification, critical infrastructure, education and vocational training, employment and worker management (incl. CV screening), access to essential services, law enforcement, migration/asylum/border control, administration of justice and democratic processes.
- LIMITED (Article 50): Transparency obligations only. Chatbots interacting with humans, AI-generated synthetic content (deepfakes), emotion recognition or biometric categorization (where not prohibited), AI-generated text on matters of public interest.
- MINIMAL: All other AI systems. No specific obligations under the AI Act. Examples: spam filters, recommender systems for non-essential services, AI in video games, inventory management.

AI SYSTEM TO CLASSIFY:
{system_description}

RELEVANT EU AI ACT EXCERPTS:
{context}

Classify this AI system. Respond ONLY with valid JSON in exactly this format, no other text:
{{
  "risk_category": "PROHIBITED" or "HIGH" or "LIMITED" or "MINIMAL",
  "confidence": <number between 0.0 and 1.0>,
  "relevant_articles": ["Article X", "Annex Y point Z"],
  "reasoning": "<2-3 sentences citing specific Annex/Article references and naming which prohibited or high-risk use case this maps to>"
}}
"""


def classify_ai_system(system_description: str, k: int = 6) -> Dict:
    """
    Classify an AI system into one of the four EU AI Act risk categories.

    Args:
        system_description: Plain-language description of the AI system.
        k: Number of EU AI Act chunks to retrieve as context.

    Returns:
        Dict with keys: risk_category, confidence, relevant_articles, reasoning, sources.
        On parse failure, returns dict with 'error' key and raw LLM response.
    """
    retrieval_query = (
        f"AI risk classification high-risk prohibited Annex III: {system_description}"
    )
    hits = retrieve(query=retrieval_query, k=k, source_filter="eu_ai_act")

    context = "\n\n".join([
        f"[{h['source']} page {h['page']}]\n{h['text']}"
        for h in hits
    ])

    prompt = CLASSIFICATION_PROMPT.format(
        system_description=system_description,
        context=context,
    )

    response_text = generate(prompt)

    # Extract JSON (LLM may wrap in markdown code fences)
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return {
            "error": "No JSON found in LLM response",
            "raw_response": response_text,
            "sources": hits,
        }

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return {
            "error": f"JSON decode error: {e}",
            "raw_response": response_text,
            "sources": hits,
        }

    # Validate category
    category = result.get("risk_category", "").upper()
    if category not in VALID_CATEGORIES:
        return {
            "error": f"Invalid risk_category: {category}. Must be one of {VALID_CATEGORIES}",
            "raw_response": response_text,
            "parsed": result,
            "sources": hits,
        }

    result["risk_category"] = category
    result["sources"] = hits
    return result
