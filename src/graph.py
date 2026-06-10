"""
LangGraph orchestration for AIActChecker.

This is the integration layer that routes user queries to the right mode.

Current status:
  - Mode 1 (Classify): WIRED, runs the real classify_ai_system function
  - Mode 2 (Obligations): STUB, returns placeholder
  - Mode 3 (Gap Analysis): STUB, returns placeholder
  - Mode 4 (Cross Map): STUB, returns placeholder

Routing:
  Keyword based intent detection. Simple, deterministic, no extra LLM call.
  When all 4 modes are wired and we want more flexible routing, we can swap
  in an LLM based intent classifier without changing the graph structure.
"""

from typing import TypedDict, Optional, Literal

from langgraph.graph import StateGraph, END

from src.classify import classify_ai_system


# ----- State model -----

class AppState(TypedDict, total=False):
    """State that flows through the graph."""
    user_input: str
    mode: Optional[str]
    system_description: Optional[str]
    current_state: Optional[str]
    topic: Optional[str]
    risk_category: Optional[str]

    # Per mode outputs
    classification: Optional[dict]
    obligations: Optional[dict]
    gap_analysis: Optional[dict]
    cross_map: Optional[dict]

    error: Optional[str]


# ----- Router -----

def router_node(state: AppState) -> AppState:
    """
    Decide which mode to run based on keywords in the user input.

    This is intentionally simple. Real keyword overlap and ambiguity will be
    handled later by an LLM intent classifier or by exposing modes as
    explicit API endpoints in the FastAPI layer.
    """
    text = state["user_input"].lower()

    if any(kw in text for kw in ["classify", "what risk", "risk category", "risk tier", "what tier"]):
        mode = "CLASSIFY"
    elif any(kw in text for kw in ["obligations", "checklist", "what must i do", "requirements"]):
        mode = "OBLIGATIONS"
    elif any(kw in text for kw in ["gap", "audit", "current state", "where am i falling"]):
        mode = "GAP_ANALYSIS"
    elif any(kw in text for kw in ["compare", "map across", "cross framework", "gdpr", "nist"]):
        mode = "CROSS_MAP"
    else:
        # Default: assume the user is describing a system and wants classification
        mode = "CLASSIFY"

    # For CLASSIFY / OBLIGATIONS / GAP_ANALYSIS the input is a system description.
    # For CROSS_MAP the input is a topic.
    if mode == "CROSS_MAP":
        return {**state, "mode": mode, "topic": state["user_input"]}
    else:
        return {**state, "mode": mode, "system_description": state["user_input"]}


# ----- Mode nodes -----

def classify_node(state: AppState) -> AppState:
    """Mode 1: Risk classification. WIRED to real function."""
    result = classify_ai_system(state["system_description"])
    return {**state, "classification": result}


def obligations_node(state: AppState) -> AppState:
    """Mode 2: Obligations checklist. STUB. Will be wired in next iteration."""
    return {
        **state,
        "obligations": {
            "status": "stub",
            "note": "Mode 2 not yet wired to the graph. Call get_obligations() directly for now.",
        },
    }


def gap_analysis_node(state: AppState) -> AppState:
    """Mode 3: Gap analysis. STUB. Will be wired in next iteration."""
    return {
        **state,
        "gap_analysis": {
            "status": "stub",
            "note": "Mode 3 not yet wired to the graph. Call analyze_gaps() directly for now.",
        },
    }


def cross_map_node(state: AppState) -> AppState:
    """Mode 4: Cross standard map. STUB. Will be wired in next iteration."""
    return {
        **state,
        "cross_map": {
            "status": "stub",
            "note": "Mode 4 not yet wired to the graph. Call cross_map() directly for now.",
        },
    }


# ----- Conditional edge: route to mode -----

def route_to_mode(state: AppState) -> Literal["classify", "obligations", "gap_analysis", "cross_map"]:
    """Pick the next node based on the mode the router selected."""
    mode = state.get("mode", "CLASSIFY")
    return {
        "CLASSIFY": "classify",
        "OBLIGATIONS": "obligations",
        "GAP_ANALYSIS": "gap_analysis",
        "CROSS_MAP": "cross_map",
    }.get(mode, "classify")


# ----- Build graph -----

def build_graph():
    """Build and compile the LangGraph state graph."""
    graph = StateGraph(AppState)

    graph.add_node("router", router_node)
    graph.add_node("classify", classify_node)
    graph.add_node("obligations", obligations_node)
    graph.add_node("gap_analysis", gap_analysis_node)
    graph.add_node("cross_map", cross_map_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_to_mode,
        {
            "classify": "classify",
            "obligations": "obligations",
            "gap_analysis": "gap_analysis",
            "cross_map": "cross_map",
        },
    )

    graph.add_edge("classify", END)
    graph.add_edge("obligations", END)
    graph.add_edge("gap_analysis", END)
    graph.add_edge("cross_map", END)

    return graph.compile()


# ----- Public entry point -----

def run(user_input: str) -> dict:
    """Run a query end to end through the graph."""
    app = build_graph()
    initial_state: AppState = {"user_input": user_input}
    return app.invoke(initial_state)
