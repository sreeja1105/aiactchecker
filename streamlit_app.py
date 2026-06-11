"""
AIActChecker — Streamlit UI.

Interactive web interface for all 4 EU AI Act compliance modes.

Run from project root:
    streamlit run streamlit_app.py

Streamlit will open the app at http://localhost:8501
"""

import sys
from pathlib import Path

# Make src/ importable when running from project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.classify import classify_ai_system
from src.obligations import get_obligations
from src.gap_analysis import analyze_gaps
from src.cross_map import cross_map


# ----- Page config -----

st.set_page_config(
    page_title="AIActChecker",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----- Cached wrappers (avoid duplicate LLM calls on identical input) -----

@st.cache_data(ttl=3600, show_spinner=False)
def cached_classify(system_description: str):
    return classify_ai_system(system_description)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_obligations(system_description: str, risk_category: str | None):
    return get_obligations(
        system_description=system_description,
        risk_category=risk_category,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_gap_analysis(system_description: str, current_state: str, risk_category: str | None):
    return analyze_gaps(
        system_description=system_description,
        current_state=current_state,
        risk_category=risk_category,
    )


@st.cache_data(ttl=3600, show_spinner=False)
def cached_cross_map(topic: str):
    return cross_map(topic)


# ----- Sidebar -----

with st.sidebar:
    st.title("⚖️ AIActChecker")
    st.caption("Multi mode EU AI Act compliance analysis")

    st.divider()

    st.markdown("### What it does")
    st.markdown(
        "Analyze AI systems against the EU AI Act, GDPR, and NIST AI RMF. "
        "Four modes: classification, obligations, gap analysis, cross framework mapping. "
        "All outputs include article level citations."
    )

    st.markdown("### Tech stack")
    st.markdown(
        "- RAG over 1671 chunks (ChromaDB)\n"
        "- Local sentence transformers (mpnet base v2)\n"
        "- Gemini 2.5 Flash Lite generation\n"
        "- LangGraph orchestration\n"
        "- FastAPI + Streamlit"
    )

    st.markdown("### Links")
    st.markdown(
        "- [GitHub](https://github.com/sreeja1105/aiactchecker)\n"
        "- [Author](https://linkedin.com/in/kotha-sreeja)"
    )

    st.divider()
    st.caption(
        "Built end to end agent first using Claude as primary coding pair. "
        "Free tier API constraints shape the architecture."
    )


# ----- Main area -----

st.title("AIActChecker")
st.caption("EU AI Act compliance analysis with article level citations")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Classify",
    "2. Obligations",
    "3. Gap Analysis",
    "4. Cross Standard Map",
])


# ----- Helper: render classification result -----

def render_classification(result: dict):
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return

    risk = result.get("risk_category", "?")

    if risk == "PROHIBITED":
        st.error(f"**Risk category: {risk}**")
    elif risk == "HIGH":
        st.warning(f"**Risk category: {risk}**")
    elif risk == "LIMITED":
        st.info(f"**Risk category: {risk}**")
    elif risk == "MINIMAL":
        st.success(f"**Risk category: {risk}**")
    else:
        st.write(f"**Risk category: {risk}**")

    refs = result.get("article_references", [])
    if refs:
        st.markdown(f"**Citations:** {', '.join(refs)}")

    reasoning = result.get("reasoning", "")
    if reasoning:
        st.markdown(f"**Reasoning:**")
        st.write(reasoning)


# ----- Tab 1: Classify -----

with tab1:
    st.subheader("Mode 1 — Risk Classification")
    st.caption("Classify an AI system into one of four EU AI Act risk tiers with article level reasoning.")

    sys_desc_1 = st.text_area(
        "Describe the AI system",
        placeholder="An AI tool that screens job applicants by scoring CVs and ranking candidates for HR teams.",
        height=120,
        key="classify_input",
    )

    if st.button("Classify", type="primary", key="classify_btn"):
        if not sys_desc_1.strip():
            st.warning("Enter a system description first.")
        else:
            with st.spinner("Classifying..."):
                result = cached_classify(sys_desc_1)
            render_classification(result)


# ----- Tab 2: Obligations -----

with tab2:
    st.subheader("Mode 2 — Obligations Checklist")
    st.caption("Generate the obligations that apply to the system, with concrete required actions.")

    sys_desc_2 = st.text_area(
        "Describe the AI system",
        placeholder="An AI tool that screens job applicants by scoring CVs...",
        height=120,
        key="obligations_input",
    )

    risk_2 = st.selectbox(
        "Pre classified risk category (optional, saves an LLM call)",
        ["(auto classify first)", "HIGH", "LIMITED", "MINIMAL", "PROHIBITED"],
        key="obligations_risk",
    )
    risk_2_value = None if risk_2 == "(auto classify first)" else risk_2

    if st.button("Generate checklist", type="primary", key="obligations_btn"):
        if not sys_desc_2.strip():
            st.warning("Enter a system description first.")
        else:
            with st.spinner("Generating obligations..."):
                result = cached_obligations(sys_desc_2, risk_2_value)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                risk = result.get("risk_category", "?")
                st.markdown(f"**Risk category:** {risk}")
                obligations_list = result.get("obligations", [])
                st.markdown(f"**{len(obligations_list)} obligations apply:**")

                for i, ob in enumerate(obligations_list, start=1):
                    title = ob.get("title", "?")
                    article = ob.get("article", "?")
                    with st.expander(f"{i}. {title}  —  [{article}]"):
                        for field_key, field_label in [
                            ("description", "Description"),
                            ("required_action", "Required action"),
                            ("rationale", "Rationale"),
                        ]:
                            if ob.get(field_key):
                                st.markdown(f"**{field_label}:** {ob[field_key]}")


# ----- Tab 3: Gap Analysis -----

with tab3:
    st.subheader("Mode 3 — Gap Analysis")
    st.caption(
        "Compare current implementation state against the obligations. "
        "Returns MET / PARTIAL / GAP status per requirement with priority and recommended actions."
    )

    sys_desc_3 = st.text_area(
        "Describe the AI system",
        placeholder="An AI tool that screens job applicants...",
        height=100,
        key="gap_sys",
    )

    current_state_3 = st.text_area(
        "Describe your current implementation and controls",
        placeholder=(
            "We use AI scores as one of several inputs to HR decisions. A human reviews "
            "top candidates. We log model outputs. We have not documented our risk management "
            "process or training data sources..."
        ),
        height=150,
        key="gap_state",
    )

    risk_3 = st.selectbox(
        "Risk category (saves an LLM call if known)",
        ["(auto classify first)", "HIGH", "LIMITED", "MINIMAL", "PROHIBITED"],
        key="gap_risk",
    )
    risk_3_value = None if risk_3 == "(auto classify first)" else risk_3

    if st.button("Run audit", type="primary", key="gap_btn"):
        if not sys_desc_3.strip() or not current_state_3.strip():
            st.warning("Both the system description and the current state are required.")
        else:
            with st.spinner("Auditing compliance..."):
                result = cached_gap_analysis(sys_desc_3, current_state_3, risk_3_value)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                summary = result.get("summary", {})
                score = summary.get("compliance_score_percent", 0)

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Compliance score", f"{score}%")
                with col2:
                    st.progress(score / 100 if isinstance(score, (int, float)) else 0)

                col_m, col_p, col_g, col_c = st.columns(4)
                col_m.metric("Met", summary.get("met", 0))
                col_p.metric("Partial", summary.get("partial", 0))
                col_g.metric("Gap", summary.get("gaps", 0))
                col_c.metric("Critical", summary.get("critical_gaps_count", 0))

                st.markdown(f"**Risk category:** {result.get('risk_category', '?')}")
                st.divider()

                status_icon = {"MET": "✅", "PARTIAL": "⚠️", "GAP": "❌"}
                priority_color = {
                    "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"
                }

                for item in result.get("gap_analysis", []):
                    icon = status_icon.get(item.get("status"), "•")
                    prio = priority_color.get(item.get("priority"), "")
                    title = item.get("requirement_title", "?")
                    article = item.get("article", "?")
                    status = item.get("status", "?")
                    priority = item.get("priority", "?")

                    with st.expander(f"{icon} {title}  —  [{article}]  ·  {status}  ·  {prio} {priority}"):
                        if item.get("current_state_assessment"):
                            st.markdown(f"**Currently:** {item['current_state_assessment']}")
                        if item.get("status") != "MET":
                            if item.get("gap_description"):
                                st.markdown(f"**Missing:** {item['gap_description']}")
                            if item.get("recommended_action"):
                                st.markdown(f"**Action:** {item['recommended_action']}")


# ----- Tab 4: Cross Standard Map -----

with tab4:
    st.subheader("Mode 4 — Cross Standard Map")
    st.caption(
        "Map a governance topic across the EU AI Act, GDPR, and NIST AI RMF. "
        "Useful when the same concept appears across multiple frameworks with different scopes."
    )

    topic_4 = st.text_input(
        "Topic",
        placeholder="Data Governance and Data Quality for AI Systems",
        key="cross_topic",
    )

    if st.button("Map topic", type="primary", key="cross_btn"):
        if not topic_4.strip():
            st.warning("Enter a topic first.")
        else:
            with st.spinner("Mapping across frameworks..."):
                result = cached_cross_map(topic_4)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.markdown(f"**Topic:** {result.get('topic', '?')}")
                st.divider()

                mappings = result.get("mappings", {})

                col_eu, col_gd, col_ni = st.columns(3)

                def render_framework(col, mapping: dict, label: str):
                    with col:
                        st.markdown(f"#### {label}")
                        refs = mapping.get("primary_references", [])
                        if refs:
                            st.markdown(f"**References:** {', '.join(refs[:5])}")
                        summary = mapping.get("summary", "")
                        if summary:
                            st.markdown(f"**Summary:**")
                            st.write(summary)
                        obligations = mapping.get("key_obligations", [])
                        if obligations:
                            st.markdown("**Key obligations:**")
                            for ob in obligations:
                                st.markdown(f"- {ob}")

                render_framework(col_eu, mappings.get("eu_ai_act", {}), "EU AI Act")
                render_framework(col_gd, mappings.get("gdpr", {}), "GDPR")
                render_framework(col_ni, mappings.get("nist_ai_rmf", {}), "NIST AI RMF")

                st.divider()

                if result.get("overlap"):
                    st.markdown("#### Overlap")
                    st.write(result["overlap"])

                if result.get("differences"):
                    st.markdown("#### Differences")
                    st.write(result["differences"])

                if result.get("compliance_guidance"):
                    st.markdown("#### Compliance guidance")
                    st.write(result["compliance_guidance"])
