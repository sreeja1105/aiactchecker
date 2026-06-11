"""
LangGraph orchestration for AIActChecker.

This is the integration layer that routes user queries to the right mode.

Current status: ALL 4 MODES WIRED.
  - Mode 1 (Classify):     classify_ai_system()
  - Mode 2 (Obligations):  get_obligations()
  - Mode 3 (Gap Analysis): analyze_gaps()
  - Mode 4 (Cross Map):    cross_map()

Routing:
  Keyword based intent detection. Simple, deterministic, no extra LLM call.
  Mode specific inputs (system_description, current_state, topic, risk_category)
  can be passed explicitly in initial state when the caller already knows them.
  Useful for Mode 3 which needs both a system description and a current state.
"""

from typing import TypedDict, Optional, Literal

from langgraph.graph import StateGraph, END

from src.classify import classify_ai_system
from src.obligations import get_obligations
from src.gap_analysis import analyze_gaps
from src.cross_map import cross_map


# ----- State model -----

class AppState(TypedDict, total=False):
    """State that flows through the graph."""
    user_input: str
    mode: Optional[str]

    # Mode specific inputs (can be supplied by caller or filled in by the router)
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

    If the caller has already supplied mode specific inputs (system_description,
    topic, current_state) in the initial state, those are preserved. Otherwise
    the router fills them from user_input as a sensible default.
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
        mode = "CLASSIFY"

    new_state: AppState = {**state, "mode": mode}

    # Fill in mode specific inputs from user_input only if the caller did not supply them.
    if mode == "CROSS_MAP":
        if not new_state.get("topic"):
            new_state["topic"] = state["user_input"]
    else:
        if not new_state.get("system_description"):
            new_state["system_description"] = state["user_input"]

    return new_state


# ----- Mode nodes (all wired) -----

def classify_node(state: AppState) -> AppState:
    """Mode 1: Risk classification."""
    result = classify_ai_system(state["system_description"])
    return {**state, "classification": result}


def obligations_node(state: AppState) -> AppState:
    """Mode 2: Obligations checklist."""
    result = get_obligations(
        system_description=state["system_description"],
        risk_category=state.get("risk_category"),
    )
    return {**state, "obligations": result}


def gap_analysis_node(state: AppState) -> AppState:
    """Mode 3: Gap analysis. Requires both system_description and current_state."""
    current_state = state.get("current_state")
    if not current_state:
        return {
            **state,
            "gap_analysis": {
                "error": "Mode 3 requires a current_state input describing the user's "
                         "existing controls. Pass it explicitly in the initial state."
            },
        }

    result = analyze_gaps(
        system_description=state["system_description"],
        current_state=current_state,
        risk_category=state.get("risk_category"),
    )
    return {**state, "gap_analysis": result}


def cross_map_node(state: AppState) -> AppState:
    """Mode 4: Cross standard map."""
    result = cross_map(state["topic"])
    return {**state, "cross_map": result}


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

def run(
    user_input: str,
    *,
    system_description: Optional[str] = None,
    current_state: Optional[str] = None,
    topic: Optional[str] = None,
    risk_category: Optional[str] = None,
) -> dict:
    """
    Run a query end to end through the graph.

    Args:
        user_input: The user's natural language query. The router uses this to
                    pick the mode.
        system_description: Optional. For Modes 1, 2, 3. If not provided, the
                            user_input is used.
        current_state: Required for Mode 3 (gap analysis). The user's current
                       compliance controls.
        topic: Optional. For Mode 4. If not provided, the user_input is used.
        risk_category: Optional. Pre classified risk tier (PROHIBITED, HIGH,
                       LIMITED, MINIMAL). Saves an LLM call in Modes 2 and 3.
    """
    app = build_graph()

    initial_state: AppState = {"user_input": user_input}
    if system_description:
        initial_state["system_description"] = system_description
    if current_state:
        initial_state["current_state"] = current_state
    if topic:
        initial_state["topic"] = topic
    if risk_category:
        initial_state["risk_category"] = risk_category

    return app.invoke(initial_state)
