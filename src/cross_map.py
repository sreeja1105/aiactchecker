"""
Mode 4: Cross-Standard Map.

Maps a governance topic across the three frameworks ingested into the corpus:
  - EU AI Act
  - GDPR
  - NIST AI Risk Management Framework

Retrieval is local (no API calls for embeddings). Only the final synthesis
hits Gemini, which is 1 LLM call per topic.
"""

import json
import re
from typing import Dict, List

from src.llm import generate
from src.retrieval import retrieve

CROSS_MAP_PROMPT = """You are an expert in AI governance with deep knowledge of the EU AI Act, GDPR, and NIST AI Risk Management Framework.

The user wants to understand how the following topic is treated across these three frameworks:

TOPIC: {topic}

RELEVANT EU AI ACT EXCERPTS:
{eu_ai_act_context}

RELEVANT GDPR EXCERPTS:
{gdpr_context}

RELEVANT NIST AI RMF EXCERPTS:
{nist_context}

Produce a structured cross-framework mapping. Respond ONLY with valid JSON in exactly this format:
{{
  "topic": "{topic}",
  "mappings": {{
    "eu_ai_act": {{
      "primary_references": ["Article X", "Article Y"],
      "summary": "1-2 sentence summary of how the EU AI Act addresses this topic",
      "key_obligations": ["Concrete obligation 1", "Concrete obligation 2"]
    }},
    "gdpr": {{
      "primary_references": ["Article X", "Recital Y"],
      "summary": "1-2 sentence summary of how GDPR addresses this topic",
      "key_obligations": ["Concrete obligation 1", "Concrete obligation 2"]
    }},
    "nist_ai_rmf": {{
      "primary_references": ["GOVERN-X.Y", "MEASURE-X.Y"],
      "summary": "1-2 sentence summary of how NIST AI RMF addresses this topic",
      "key_obligations": ["Concrete control 1", "Concrete control 2"]
    }}
  }},
  "overlap": "Where the three frameworks align in scope or requirements (2-3 sentences)",
  "differences": "Where they diverge (e.g., GDPR focuses on personal data only, while AI Act covers all training data; NIST is voluntary while EU AI Act is binding law)",
  "compliance_guidance": "Practical guidance: which is strictest, does complying with one cover the others, what should the user prioritize"
}}

If a framework does not address the topic, state that explicitly in its summary and leave primary_references as an empty array.
"""


def cross_map(topic: str, k: int = 5) -> Dict:
    """
    Map a governance topic across EU AI Act, GDPR, and NIST AI RMF.

    Args:
        topic: The topic to map (e.g., "Data Governance", "Human Oversight").
        k: Number of chunks to retrieve from each framework.
    """
    eu_hits = retrieve(topic, k=k, source_filter="eu_ai_act")
    gdpr_hits = retrieve(topic, k=k, source_filter="gdpr")
    nist_hits = retrieve(topic, k=k, source_filter="nist_ai_rmf")

    def fmt(hits: List[Dict]) -> str:
        if not hits:
            return "No relevant content found in this corpus."
        return "\n\n".join([
            f"[page {h['page']}, similarity {h['score']}]\n{h['text']}"
            for h in hits
        ])

    prompt = CROSS_MAP_PROMPT.format(
        topic=topic,
        eu_ai_act_context=fmt(eu_hits),
        gdpr_context=fmt(gdpr_hits),
        nist_context=fmt(nist_hits),
    )

    response_text = generate(prompt)

    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if not json_match:
        return {"error": "No JSON in LLM response", "raw_response": response_text}

    try:
        result = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}", "raw_response": response_text}

    # Attach top retrieval sources for traceability
    result["retrieval_sources"] = {
        "eu_ai_act": [{"page": h["page"], "score": h["score"]} for h in eu_hits[:3]],
        "gdpr": [{"page": h["page"], "score": h["score"]} for h in gdpr_hits[:3]],
        "nist_ai_rmf": [{"page": h["page"], "score": h["score"]} for h in nist_hits[:3]],
    }

    return result
