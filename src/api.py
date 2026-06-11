"""
FastAPI service for AIActChecker.

Exposes the 4 compliance modes as REST endpoints, plus a universal /query
endpoint that uses LangGraph orchestration to route to the right mode.

Run from project root:
    uvicorn src.api:app --reload

Then visit http://localhost:8000/docs for the auto generated Swagger UI.
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.classify import classify_ai_system
from src.obligations import get_obligations
from src.gap_analysis import analyze_gaps
from src.cross_map import cross_map
from src.graph import run as graph_run


# ----- App setup -----

app = FastAPI(
    title="AIActChecker API",
    description=(
        "Multi mode EU AI Act compliance analysis service.\n\n"
        "Classify AI systems by risk tier, generate the obligations checklist, "
        "audit current compliance with prioritized gaps, and map requirements "
        "across the EU AI Act, GDPR, and NIST AI RMF.\n\n"
        "Each endpoint corresponds to one mode. The /query endpoint is universal "
        "and uses a LangGraph router to pick the right mode based on the input."
    ),
    version="0.1.0",
)

# CORS for development. Restrict allow_origins in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----- Request models -----

class ClassifyRequest(BaseModel):
    system_description: str = Field(
        ...,
        description="Plain language description of the AI system to classify.",
        json_schema_extra={
            "example": "An AI tool that screens job applicants by scoring CVs and ranking candidates for HR teams."
        },
    )


class ObligationsRequest(BaseModel):
    system_description: str = Field(
        ...,
        description="Plain language description of the AI system.",
    )
    risk_category: Optional[str] = Field(
        None,
        description=(
            "Optional pre classified risk tier (PROHIBITED, HIGH, LIMITED, MINIMAL). "
            "If omitted, classification runs first."
        ),
    )


class GapAnalysisRequest(BaseModel):
    system_description: str = Field(
        ...,
        description="Plain language description of the AI system.",
    )
    current_state: str = Field(
        ...,
        description="Description of the user's existing compliance controls.",
    )
    risk_category: Optional[str] = Field(
        None,
        description=(
            "Optional pre classified risk tier. Saves an LLM call if known."
        ),
    )


class CrossMapRequest(BaseModel):
    topic: str = Field(
        ...,
        description="Governance topic to map across frameworks.",
        json_schema_extra={
            "example": "Data Governance and Data Quality for AI Systems"
        },
    )


class QueryRequest(BaseModel):
    user_input: str = Field(
        ...,
        description=(
            "Natural language query. The LangGraph router picks the right mode "
            "based on keywords in the input."
        ),
    )
    system_description: Optional[str] = None
    current_state: Optional[str] = None
    topic: Optional[str] = None
    risk_category: Optional[str] = None


# ----- Endpoints -----

@app.get("/", tags=["meta"])
def root():
    """Service information and available endpoints."""
    return {
        "name": "AIActChecker API",
        "version": "0.1.0",
        "description": "Multi mode EU AI Act compliance analysis",
        "modes": {
            "POST /classify": "Mode 1 — risk tier classification",
            "POST /obligations": "Mode 2 — obligations checklist",
            "POST /gap-analysis": "Mode 3 — current state audit with priorities",
            "POST /cross-map": "Mode 4 — cross framework mapping",
            "POST /query": "Universal endpoint, LangGraph routed",
        },
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", tags=["meta"])
def health():
    """Simple liveness probe."""
    return {"status": "ok"}


@app.post("/classify", tags=["modes"])
def classify_endpoint(req: ClassifyRequest):
    """
    Mode 1 — Risk Classification.

    Classify an AI system into one of four EU AI Act risk tiers
    (PROHIBITED, HIGH, LIMITED, MINIMAL) with article level reasoning.
    """
    try:
        return classify_ai_system(req.system_description)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {e}")


@app.post("/obligations", tags=["modes"])
def obligations_endpoint(req: ObligationsRequest):
    """
    Mode 2 — Obligations Checklist.

    Generate the full list of obligations that apply to the given AI system,
    with concrete required actions per item.
    """
    try:
        return get_obligations(
            system_description=req.system_description,
            risk_category=req.risk_category,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Obligations failed: {e}")


@app.post("/gap-analysis", tags=["modes"])
def gap_analysis_endpoint(req: GapAnalysisRequest):
    """
    Mode 3 — Gap Analysis.

    Compare the user's current implementation against the obligations for the
    system's risk tier. Returns per requirement status (MET, PARTIAL, GAP),
    priority (CRITICAL, HIGH, MEDIUM, LOW), and recommended actions.
    """
    try:
        return analyze_gaps(
            system_description=req.system_description,
            current_state=req.current_state,
            risk_category=req.risk_category,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gap analysis failed: {e}")


@app.post("/cross-map", tags=["modes"])
def cross_map_endpoint(req: CrossMapRequest):
    """
    Mode 4 — Cross Standard Map.

    Map a governance topic (for example data governance, human oversight)
    across the EU AI Act, GDPR, and NIST AI RMF. Returns references,
    summaries, overlap, differences, and practical compliance guidance.
    """
    try:
        return cross_map(req.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cross map failed: {e}")


@app.post("/query", tags=["universal"])
def query_endpoint(req: QueryRequest):
    """
    Universal endpoint.

    Routes the user input to the appropriate mode via the LangGraph orchestrator.
    Useful when the caller does not know which mode to invoke.

    Mode specific inputs (system_description, current_state, topic, risk_category)
    can be supplied explicitly when known.
    """
    try:
        return graph_run(
            user_input=req.user_input,
            system_description=req.system_description,
            current_state=req.current_state,
            topic=req.topic,
            risk_category=req.risk_category,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
