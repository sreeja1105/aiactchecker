"""
AIActChecker - Streamlit UI (final redesigned version with charts).

Design philosophy: guide non-experts, do not report data at analysts.
Each result interprets first, shows supporting data second, gives next actions third.

Run from project root:
    streamlit run streamlit_app.py
"""

import re
import sys
from collections import Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import pypdf
import streamlit as st

from src.classify import classify_ai_system
from src.obligations import get_obligations
from src.gap_analysis import analyze_gaps
from src.cross_map import cross_map

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether
)


# ----- Page config -----

st.set_page_config(
    page_title="AIActChecker",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----- Design system (CSS) -----

st.markdown("""
<style>
:root {
    --ai-text-primary: #0F172A;
    --ai-text-secondary: #475569;
    --ai-text-tertiary: #94A3B8;
    --ai-bg-secondary: #F8FAFC;
    --ai-bg-tertiary: #F1F5F9;
    --ai-border: #E2E8F0;
    --ai-accent: #1E293B;
}

.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }
/* Tabs styled as clickable buttons */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: none;
    background: transparent;
    margin-bottom: 16px;
}

.stTabs [data-baseweb="tab"] {
    padding: 9px 18px;
    font-size: 13px;
    font-weight: 500;
    background: var(--ai-bg-secondary);
    border: 1px solid var(--ai-border);
    border-radius: 6px;
    color: var(--ai-text-secondary);
    transition: all 0.15s ease;
    height: auto;
}

.stTabs [data-baseweb="tab"]:hover {
    background: var(--ai-bg-tertiary);
    color: var(--ai-text-primary);
    border-color: #CBD5E1;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--ai-text-primary);
    color: white;
    border-color: var(--ai-text-primary);
}

.stTabs [data-baseweb="tab"][aria-selected="true"]:hover {
    background: #1E293B;
    color: white;
}

/* Hide the default red underline indicator */
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none;
}
h1 { font-weight: 600; letter-spacing: -0.5px; color: var(--ai-text-primary); }
h2 { font-weight: 600; color: var(--ai-text-primary); }
h3 { font-weight: 500; color: var(--ai-text-primary); margin-top: 0.5rem; }

.ai-subtitle { color: var(--ai-text-secondary); margin-top: -8px; margin-bottom: 24px; font-size: 15px; }
.ai-eyebrow { font-size: 11px; color: var(--ai-text-tertiary); letter-spacing: 0.5px; text-transform: uppercase; font-weight: 500; margin-bottom: 6px; }

.ai-risk-card { border: 1px solid var(--ai-border); border-radius: 8px; padding: 20px 24px; margin-bottom: 16px; border-left-width: 3px; }
.ai-risk-card.prohibited { border-left-color: #B91C1C; background: #FEF2F2; }
.ai-risk-card.high       { border-left-color: #C2410C; background: #FFF7ED; }
.ai-risk-card.limited    { border-left-color: #CA8A04; background: #FEFCE8; }
.ai-risk-card.minimal    { border-left-color: #15803D; background: #F0FDF4; }

.ai-risk-card .risk-label { font-size: 11px; letter-spacing: 0.5px; text-transform: uppercase; font-weight: 500; margin-bottom: 4px; }
.ai-risk-card.prohibited .risk-label { color: #991B1B; }
.ai-risk-card.high .risk-label       { color: #9A3412; }
.ai-risk-card.limited .risk-label    { color: #854D0E; }
.ai-risk-card.minimal .risk-label    { color: #166534; }

.ai-risk-card .risk-tier { font-size: 22px; font-weight: 600; margin-bottom: 8px; }
.ai-risk-card .system-name { font-size: 14px; color: var(--ai-text-secondary); line-height: 1.55; }

.ai-callout { background: var(--ai-bg-secondary); border-radius: 6px; padding: 14px 18px; margin: 12px 0; }
.ai-callout-title { font-size: 11px; color: var(--ai-text-tertiary); letter-spacing: 0.5px; text-transform: uppercase; font-weight: 500; margin-bottom: 6px; }
.ai-callout-body { font-size: 14px; line-height: 1.6; color: var(--ai-text-primary); }

.ai-citation-card { background: white; border: 1px solid var(--ai-border); border-radius: 6px; padding: 12px 14px; height: 100%; }
.ai-citation-card .label { font-size: 11px; color: var(--ai-text-tertiary); letter-spacing: 0.4px; text-transform: uppercase; margin-bottom: 4px; }
.ai-citation-card .ref { font-size: 14px; font-weight: 600; margin-bottom: 4px; color: var(--ai-text-primary); }
.ai-citation-card .desc { font-size: 12px; color: var(--ai-text-secondary); line-height: 1.45; }

.ai-action { background: var(--ai-bg-secondary); border-radius: 6px; padding: 14px 16px; margin-bottom: 8px; display: flex; gap: 14px; align-items: flex-start; }
.ai-action-num { flex-shrink: 0; width: 24px; height: 24px; background: #991B1B; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: 600; }
.ai-action-num.high { background: #9A3412; }
.ai-action-num.medium { background: #3730A3; }
.ai-action-content { flex: 1; }
.ai-action-title { font-size: 14px; font-weight: 500; margin-bottom: 3px; color: var(--ai-text-primary); }
.ai-action-meta { font-size: 12px; color: var(--ai-text-tertiary); margin-bottom: 6px; }
.ai-action-desc { font-size: 13px; color: var(--ai-text-secondary); line-height: 1.5; }

.ai-badge { display: inline-block; padding: 2px 9px; border-radius: 4px; font-size: 11px; font-weight: 600; letter-spacing: 0.3px; line-height: 1.5; }
.ai-badge.gap, .ai-badge.critical { background: #FEE2E2; color: #991B1B; }
.ai-badge.partial, .ai-badge.high { background: #FEF3C7; color: #92400E; }
.ai-badge.met                     { background: #DCFCE7; color: #166534; }
.ai-badge.medium                  { background: #E0E7FF; color: #3730A3; }
.ai-badge.low                     { background: #F1F5F9; color: #475569; }
.ai-badge.binding                 { background: #E0E7FF; color: #3730A3; }
.ai-badge.voluntary               { background: #F1F5F9; color: #64748B; }

.ai-framework-card { background: white; border: 1px solid var(--ai-border); border-radius: 8px; padding: 16px; height: 100%; }
.ai-framework-name { font-size: 15px; font-weight: 600; margin: 8px 0 12px; color: var(--ai-text-primary); }
.ai-framework-section { margin-top: 10px; }
.ai-framework-label { font-size: 10px; color: var(--ai-text-tertiary); letter-spacing: 0.4px; text-transform: uppercase; font-weight: 500; margin-bottom: 3px; }
.ai-framework-content { font-size: 12px; color: var(--ai-text-secondary); line-height: 1.5; }

.ai-section-header { font-size: 16px; font-weight: 600; margin: 24px 0 12px; color: var(--ai-text-primary); }

.ai-chart-card { background: white; border: 1px solid var(--ai-border); border-radius: 8px; padding: 16px 18px; }
.ai-chart-title { font-size: 13px; color: var(--ai-text-secondary); margin-bottom: 8px; font-weight: 500; }

/* Obligations table */
.ai-obligations-table { width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px; border: 1px solid var(--ai-border); border-radius: 6px; overflow: hidden; }
.ai-obligations-table thead { background: var(--ai-bg-secondary); }
.ai-obligations-table th { text-align: left; padding: 10px 12px; font-weight: 600; color: var(--ai-text-secondary); font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px; border-bottom: 1px solid var(--ai-border); }
.ai-obligations-table td { padding: 12px; border-bottom: 1px solid var(--ai-border); vertical-align: top; line-height: 1.55; color: var(--ai-text-primary); }
.ai-obligations-table tr:last-child td { border-bottom: none; }
.ai-obligations-table .col-num     { width: 36px; color: var(--ai-text-tertiary); text-align: center; }
.ai-obligations-table .col-title   { width: 22%; font-weight: 600; }
.ai-obligations-table .col-article { width: 90px; color: var(--ai-text-secondary); white-space: nowrap; }
.ai-obligations-table .col-action  { color: var(--ai-text-secondary); }

section[data-testid="stSidebar"] { background: var(--ai-bg-secondary); }
section[data-testid="stSidebar"] hr { margin: 14px 0; border-color: var(--ai-border); }
</style>
""", unsafe_allow_html=True)


# ----- Cached wrappers -----

@st.cache_data(ttl=3600, show_spinner=False)
def cached_classify(system_description: str):
    return classify_ai_system(system_description)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_obligations(system_description: str, risk_category):
    return get_obligations(system_description=system_description, risk_category=risk_category)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_gap_analysis(system_description: str, current_state: str, risk_category):
    return analyze_gaps(system_description=system_description, current_state=current_state, risk_category=risk_category)

@st.cache_data(ttl=3600, show_spinner=False)
def cached_cross_map(topic: str):
    return cross_map(topic)


# ----- Helpers -----

def extract_pdf_text(uploaded_file) -> str:
    try:
        reader = pypdf.PdfReader(BytesIO(uploaded_file.read()))
        parts = [(p.extract_text() or "").strip() for p in reader.pages]
        return "\n\n".join([p for p in parts if p])
    except Exception as e:
        return f"[error extracting PDF: {e}]"


def sanitize_references(refs):
    """Filter out RAG retrieval metadata (page numbers, similarity scores).
    Keep only proper article/recital references."""
    if not refs:
        return []
    cleaned = []
    skip_patterns = [
        r"similarity",
        r"^page\s+\d+",
        r"chunk[\s_]\d+",
        r"score:?\s*-?\d",
    ]
    skip_regex = re.compile("|".join(skip_patterns), re.IGNORECASE)
    for ref in refs:
        if not isinstance(ref, str):
            continue
        ref_clean = ref.strip()
        if not ref_clean:
            continue
        if skip_regex.search(ref_clean):
            continue
        # If a reference has comma-separated parts, filter each
        parts = [p.strip() for p in ref_clean.split(",")]
        good_parts = [p for p in parts if not skip_regex.search(p) and p]
        if good_parts:
            cleaned.append(", ".join(good_parts))
    return cleaned[:5]


def render_risk_card(risk_category: str, system_description: str, interpretation: str):
    risk_class_map = {"PROHIBITED": "prohibited", "HIGH": "high", "LIMITED": "limited", "MINIMAL": "minimal"}
    label_map = {"PROHIBITED": "Prohibited", "HIGH": "High risk", "LIMITED": "Limited risk", "MINIMAL": "Minimal risk"}
    risk_class = risk_class_map.get(risk_category, "minimal")
    risk_label = label_map.get(risk_category, risk_category)

    st.markdown(
        f'<div class="ai-risk-card {risk_class}">'
        f'<div class="risk-label">EU AI Act classification</div>'
        f'<div class="risk-tier">{risk_label}</div>'
        f'<div class="system-name">{system_description}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if interpretation:
        st.markdown(
            f'<div class="ai-callout">'
            f'<div class="ai-callout-title">What this means for you</div>'
            f'<div class="ai-callout-body">{interpretation}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def interpretation_for_risk(risk_category: str, obligations_count: int = None) -> str:
    if risk_category == "PROHIBITED":
        return ("Article 5 of the EU AI Act prohibits this category of AI system. "
                "It cannot be placed on the EU market or put into service, regardless of safeguards.")
    if risk_category == "HIGH":
        count = f"{obligations_count} specific obligations" if obligations_count else "13 specific obligations"
        return (f"Before this system can be placed on the EU market, you need to satisfy {count} from the EU AI Act, "
                "complete a conformity assessment, register the system in the EU AI database, and obtain a CE marking.")
    if risk_category == "LIMITED":
        return ("The system is allowed, but Article 50 requires you to be transparent with users that they are "
                "interacting with AI or seeing AI generated content. Mark generated content and disclose AI use.")
    if risk_category == "MINIMAL":
        return ("No specific EU AI Act obligations apply, though Article 95 encourages voluntary codes of conduct. "
                "Best practice is still to document the system and monitor it in production.")
    return ""


def render_obligations_table(obligations_list: list):
    rows_html = []
    for i, ob in enumerate(obligations_list, start=1):
        title = ob.get("title", "?")
        article = ob.get("article", "?")
        action = ob.get("required_action") or ob.get("description") or ""
        rows_html.append(
            f'<tr>'
            f'<td class="col-num">{i}</td>'
            f'<td class="col-title">{title}</td>'
            f'<td class="col-article">{article}</td>'
            f'<td class="col-action">{action}</td>'
            f'</tr>'
        )
    table_html = (
        '<table class="ai-obligations-table">'
        '<thead><tr><th class="col-num">#</th><th class="col-title">Obligation</th><th class="col-article">Article</th><th class="col-action">What you need to do</th></tr></thead>'
        '<tbody>' + "".join(rows_html) + '</tbody>'
        '</table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


# ----- Chart helpers -----

def make_status_donut(met: int, partial: int, gaps: int):
    """Donut chart showing Met / Partial / Gap proportions."""
    labels = ["Met", "Partial", "Gap"]
    values = [met, partial, gaps]
    color_map = ["#15803D", "#9A3412", "#991B1B"]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.62,
        marker=dict(colors=color_map, line=dict(color="white", width=2)),
        textinfo="value", textposition="inside",
        textfont=dict(size=14, color="white", family="Helvetica"),
        hovertemplate="<b>%{label}</b>: %{value}<extra></extra>",
        sort=False,
    )])
    total = sum(values) or 1
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="v", x=1.0, y=0.5, font=dict(size=12, color="#475569"), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=20, t=10, b=10),
        height=240,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"<b>{total}</b><br><span style='font-size:11px;color:#94A3B8'>TOTAL</span>",
                          x=0.5, y=0.5, font=dict(size=18, color="#0F172A"), showarrow=False)],
    )
    return fig


def make_priority_bars(gap_items: list):
    """Horizontal bar chart showing priority distribution."""
    counts = Counter(it.get("priority", "Unknown").upper() for it in gap_items)
    priorities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    counts_ordered = [counts.get(p, 0) for p in priorities]
    color_map = ["#991B1B", "#9A3412", "#3730A3", "#64748B"]
    labels = [p.title() for p in priorities]

    fig = go.Figure(data=[go.Bar(
        y=labels, x=counts_ordered, orientation="h",
        marker_color=color_map,
        text=counts_ordered, textposition="outside",
        textfont=dict(size=12, color="#0F172A", family="Helvetica"),
        hovertemplate="<b>%{y}</b>: %{x}<extra></extra>",
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=30, t=10, b=10),
        height=240,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, max(counts_ordered + [1]) * 1.25]),
        yaxis=dict(showgrid=False, autorange="reversed", tickfont=dict(size=12, color="#475569", family="Helvetica")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def make_compliance_gauge(score: int):
    """Half-donut gauge showing compliance percentage."""
    if score < 30:
        color = "#991B1B"
    elif score < 60:
        color = "#9A3412"
    else:
        color = "#15803D"

    fig = go.Figure(data=[go.Pie(
        labels=["Compliant", "Not yet"],
        values=[score, max(0, 100 - score)],
        hole=0.7,
        marker=dict(colors=[color, "#F1F5F9"], line=dict(width=0)),
        textinfo="none",
        hoverinfo="skip",
        sort=False,
        rotation=180,
        direction="clockwise",
    )])
    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        height=180,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text=f"<b style='color:{color}'>{score}%</b><br><span style='font-size:10px;color:#94A3B8;letter-spacing:0.5px'>COMPLIANT</span>",
                          x=0.5, y=0.5, font=dict(size=26), showarrow=False)],
    )
    return fig


# ----- Report generators -----

def generate_gap_report_markdown(result: dict, system_description: str, current_state: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    summary = result.get("summary", {})
    md = f"""# EU AI Act Gap Analysis Report

**Generated:** {timestamp}
**Tool:** AIActChecker · github.com/sreeja1105/aiactchecker · aiactchecker.streamlit.app

---

## System Under Review

{system_description}

## Current Implementation State

{current_state}

## Classification

**Risk Category:** {result.get('risk_category', 'Unknown')}

## Compliance Summary

| Metric | Value |
|---|---|
| Compliance Score | {summary.get('compliance_score_percent', 0)}% |
| Total Requirements | {summary.get('total_requirements', '?')} |
| Met | {summary.get('met', 0)} |
| Partial | {summary.get('partial', 0)} |
| Gap | {summary.get('gaps', 0)} |
| Critical Gaps | {summary.get('critical_gaps_count', 0)} |

---

## Detailed Findings

"""
    for i, item in enumerate(result.get("gap_analysis", []), start=1):
        md += f"### {i}. {item.get('requirement_title', '?')}  ({item.get('article', '?')})\n\n"
        md += f"- **Status:** {item.get('status', '?')}\n"
        md += f"- **Priority:** {item.get('priority', '?')}\n"
        if item.get("current_state_assessment"):
            md += f"- **Current state:** {item['current_state_assessment']}\n"
        if item.get("status") != "MET":
            if item.get("gap_description"):
                md += f"- **Missing:** {item['gap_description']}\n"
            if item.get("recommended_action"):
                md += f"- **Recommended action:** {item['recommended_action']}\n"
        md += "\n"
    md += "---\n\n*This report is generated by AIActChecker, a research tool. It is not legal advice.*\n"
    return md


def generate_gap_report_pdf(result: dict, system_description: str, current_state: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=1.8 * cm, bottomMargin=1.8 * cm, leftMargin=2 * cm, rightMargin=2 * cm)

    text_primary = colors.HexColor("#0F172A")
    text_secondary = colors.HexColor("#475569")
    text_tertiary = colors.HexColor("#94A3B8")
    border_color = colors.HexColor("#E2E8F0")
    bg_secondary = colors.HexColor("#F8FAFC")
    accent_red = colors.HexColor("#991B1B")
    accent_orange = colors.HexColor("#9A3412")
    accent_green = colors.HexColor("#166534")

    title_style = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=18, textColor=text_primary, spaceAfter=4, leading=22)
    subtitle_style = ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=10, textColor=text_tertiary, spaceAfter=16, leading=14)
    h2_style = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=13, textColor=text_primary, spaceBefore=14, spaceAfter=8, leading=16)
    body_style = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, textColor=text_primary, spaceAfter=6, leading=14)
    body_secondary_style = ParagraphStyle("BodySec", fontName="Helvetica", fontSize=10, textColor=text_secondary, spaceAfter=4, leading=14)
    footer_style = ParagraphStyle("Footer", fontName="Helvetica-Oblique", fontSize=8, textColor=text_tertiary, spaceAfter=2, leading=10, alignment=1)

    summary = result.get("summary", {})
    score = summary.get("compliance_score_percent", 0)
    risk = result.get("risk_category", "?")
    critical_count = summary.get("critical_gaps_count", 0)
    met = summary.get("met", 0)
    partial = summary.get("partial", 0)
    gaps = summary.get("gaps", 0)
    total_reqs = summary.get("total_requirements", "?")
    risk_label = {"PROHIBITED": "Prohibited", "HIGH": "High Risk", "LIMITED": "Limited Risk", "MINIMAL": "Minimal Risk"}.get(risk, risk)
    score_color = accent_red if score < 30 else (accent_orange if score < 60 else accent_green)

    story = []
    story.append(Paragraph("EU AI Act Gap Analysis Report", title_style))
    timestamp = datetime.now().strftime("%B %d, %Y at %H:%M")
    story.append(Paragraph(f"Generated: {timestamp}<br/>Tool: AIActChecker &middot; aiactchecker.streamlit.app", subtitle_style))

    score_para = Paragraph(f'<para alignment="center"><font size="22" color="{score_color.hexval()}"><b>{score}%</b></font><br/><font size="8" color="{text_tertiary.hexval()}">COMPLIANT TODAY</font></para>', body_style)
    risk_para = Paragraph(f'<para alignment="center"><font size="14"><b>{risk_label}</b></font><br/><font size="8" color="{text_tertiary.hexval()}">RISK CATEGORY</font></para>', body_style)
    critical_para = Paragraph(f'<para alignment="center"><font size="22"><b>{critical_count}</b></font><br/><font size="8" color="{text_tertiary.hexval()}">CRITICAL GAPS</font></para>', body_style)
    requirements_para = Paragraph(f'<para alignment="center"><font size="22"><b>{total_reqs}</b></font><br/><font size="8" color="{text_tertiary.hexval()}">TOTAL REQUIREMENTS</font></para>', body_style)
    summary_table = Table([[score_para, risk_para, critical_para, requirements_para]], colWidths=[4 * cm, 4 * cm, 4 * cm, 4 * cm])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg_secondary), ("BOX", (0, 0), (-1, -1), 0.5, border_color),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color), ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8), ("TOPPADDING", (0, 0), (-1, -1), 14),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 14), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 14))

    breakdown_table = Table([[
        Paragraph(f'<para alignment="center"><font size="14"><b>{met}</b></font><br/><font size="8" color="{text_tertiary.hexval()}">MET</font></para>', body_style),
        Paragraph(f'<para alignment="center"><font size="14"><b>{partial}</b></font><br/><font size="8" color="{text_tertiary.hexval()}">PARTIAL</font></para>', body_style),
        Paragraph(f'<para alignment="center"><font size="14"><b>{gaps}</b></font><br/><font size="8" color="{text_tertiary.hexval()}">GAP</font></para>', body_style),
    ]], colWidths=[5.3 * cm, 5.3 * cm, 5.3 * cm])
    breakdown_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, border_color), ("INNERGRID", (0, 0), (-1, -1), 0.5, border_color),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 10), ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(breakdown_table)
    story.append(Spacer(1, 16))

    story.append(Paragraph("System Under Review", h2_style))
    story.append(Paragraph(system_description.replace("\n", "<br/>"), body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Current Implementation State", h2_style))
    state_preview = current_state[:2500] + ("..." if len(current_state) > 2500 else "")
    story.append(Paragraph(state_preview.replace("\n", "<br/>"), body_secondary_style))
    story.append(Spacer(1, 6))

    gap_items = result.get("gap_analysis", [])
    critical_items = [it for it in gap_items if it.get("priority") == "CRITICAL" and it.get("status") in ("GAP", "PARTIAL")][:3]
    if critical_items:
        story.append(Paragraph("Start Here This Week", h2_style))
        story.append(Paragraph(f"Top {len(critical_items)} highest impact actions.", body_secondary_style))
        for i, item in enumerate(critical_items, start=1):
            title = item.get("requirement_title", "?")
            article = item.get("article", "?")
            action = item.get("recommended_action", "")
            block = [
                Paragraph(f"<b>{i}. {title}</b> &middot; <font color='{text_tertiary.hexval()}'>{article}</font>", body_style),
                Paragraph(action, body_secondary_style),
                Spacer(1, 4),
            ]
            story.append(KeepTogether(block))
        story.append(Spacer(1, 8))

    story.append(PageBreak())
    story.append(Paragraph("Full Audit Findings", h2_style))
    story.append(Paragraph(f"All {len(gap_items)} obligations evaluated against the EU AI Act.", body_secondary_style))
    story.append(Spacer(1, 6))
    for i, item in enumerate(gap_items, start=1):
        title = item.get("requirement_title", "?")
        article = item.get("article", "?")
        status = item.get("status", "?").title()
        priority = item.get("priority", "?").title()
        current = item.get("current_state_assessment", "")
        missing = item.get("gap_description", "")
        action = item.get("recommended_action", "")
        block_content = []
        block_content.append(Paragraph(f"<b>{i}. {title}</b> &middot; <font color='{text_tertiary.hexval()}'>{article} &middot; Status: {status} &middot; Priority: {priority}</font>", body_style))
        if current:
            block_content.append(Paragraph(f"<b>Currently:</b> {current}", body_secondary_style))
        if missing and status != "Met":
            block_content.append(Paragraph(f"<b>Missing:</b> {missing}", body_secondary_style))
        if action and status != "Met":
            block_content.append(Paragraph(f"<b>Recommended action:</b> {action}", body_secondary_style))
        block_content.append(Spacer(1, 8))
        story.append(KeepTogether(block_content))

    story.append(Spacer(1, 20))
    story.append(Paragraph("This report is generated by AIActChecker, a research tool. It is not legal advice. Always consult qualified counsel for compliance decisions.", footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def format_findings_dataframe(gap_analysis_items: list) -> pd.DataFrame:
    rows = []
    for item in gap_analysis_items:
        rows.append({
            "Requirement": item.get("requirement_title", "?"),
            "Article": item.get("article", "?"),
            "Status": item.get("status", "?").title(),
            "Priority": item.get("priority", "?").title(),
            "Currently": item.get("current_state_assessment", ""),
            "Missing": item.get("gap_description", "") if item.get("status") != "MET" else "",
            "Action": item.get("recommended_action", "") if item.get("status") != "MET" else "",
        })
    return pd.DataFrame(rows)


# ----- Sidebar -----

with st.sidebar:
    st.markdown("## AIActChecker")
    st.caption("EU AI Act compliance analysis")
    st.divider()
    st.markdown("**About**")
    st.markdown("Analyze AI systems against the EU AI Act, GDPR, and NIST AI RMF using RAG over 1671 chunks of regulatory text. All outputs include article level citations.")
    st.markdown("**Stack**")
    st.markdown("ChromaDB · sentence transformers · Gemini · LangGraph · FastAPI · Streamlit")
    st.divider()
    st.markdown("[GitHub Repository](https://github.com/sreeja1105/aiactchecker)  \n[Author](https://linkedin.com/in/kotha-sreeja)")
    st.divider()
    st.caption("Research tool. Not legal advice. Based on EU AI Act as adopted August 2024, with amendments through 2026.")


# ----- Main -----

st.title("AIActChecker")
st.markdown('<p class="ai-subtitle">EU AI Act compliance for any AI system. Get classification, obligations, gap analysis, and cross framework mapping in seconds.</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Risk classification", "Obligations", "Gap analysis", "Cross standard map",
])


# ============================================================================
# TAB 1 — Risk Classification
# ============================================================================

with tab1:
    st.markdown('<p class="ai-eyebrow">Mode 1</p><h3>Find out which EU AI Act risk tier applies to your system</h3>', unsafe_allow_html=True)
    sys_desc_1 = st.text_area(
        "Describe your AI system in plain language",
        placeholder="An AI tool that screens job applicants by scoring CVs and ranking candidates for HR teams.",
        height=120, key="classify_input",
    )
    col_btn, _ = st.columns([1, 4])
    with col_btn:
        run_classify = st.button("Classify", type="primary", key="classify_btn", use_container_width=True)
    if run_classify and sys_desc_1.strip():
        with st.spinner("Analyzing the system against the EU AI Act..."):
            result = cached_classify(sys_desc_1)
        st.session_state["classify_result"] = result
        st.session_state["classify_input_used"] = sys_desc_1
    if "classify_result" in st.session_state:
        result = st.session_state["classify_result"]
        if "error" in result:
            st.error(f"Could not classify. {result['error']}")
        else:
            risk = result.get("risk_category", "?")
            system_text = st.session_state.get("classify_input_used", sys_desc_1)
            render_risk_card(risk, system_text, interpretation_for_risk(risk))
            refs = sanitize_references(result.get("article_references", []))
            reasoning = result.get("reasoning", "")
            if refs or reasoning:
                st.markdown('<div class="ai-section-header">Why this classification</div>', unsafe_allow_html=True)
                if refs:
                    cols = st.columns(min(len(refs), 3))
                    for i, ref in enumerate(refs[:3]):
                        with cols[i]:
                            st.markdown(
                                f'<div class="ai-citation-card">'
                                f'<div class="label">{"Primary citation" if i == 0 else "Supporting article"}</div>'
                                f'<div class="ref">{ref}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                if reasoning:
                    st.markdown("")
                    with st.expander("See full reasoning"):
                        st.write(reasoning)


# ============================================================================
# TAB 2 — Obligations
# ============================================================================

with tab2:
    st.markdown('<p class="ai-eyebrow">Mode 2</p><h3>Generate the obligations checklist for your system</h3>', unsafe_allow_html=True)
    sys_desc_2 = st.text_area("Describe your AI system", placeholder="An AI tool that screens job applicants...", height=100, key="obligations_input")
    col_risk, col_btn = st.columns([2, 1])
    with col_risk:
        risk_2 = st.selectbox("Pre classified risk (optional, saves one LLM call)",
                              ["Auto classify first", "HIGH", "LIMITED", "MINIMAL", "PROHIBITED"], key="obligations_risk")
    risk_2_value = None if risk_2 == "Auto classify first" else risk_2
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_obligations = st.button("Generate checklist", type="primary", key="obligations_btn", use_container_width=True)
    if run_obligations and sys_desc_2.strip():
        with st.spinner("Generating obligations..."):
            result = cached_obligations(sys_desc_2, risk_2_value)
        st.session_state["obligations_result"] = result
        st.session_state["obligations_input_used"] = sys_desc_2
    if "obligations_result" in st.session_state:
        result = st.session_state["obligations_result"]
        if "error" in result:
            st.error(f"Could not generate obligations. {result['error']}")
        else:
            obligations_list = result.get("obligations", [])
            risk = result.get("risk_category", "?")
            system_text = st.session_state.get("obligations_input_used", sys_desc_2)
            render_risk_card(risk, system_text, interpretation_for_risk(risk, len(obligations_list)))
            if obligations_list:
                st.markdown('<div class="ai-section-header">Your obligations checklist</div>', unsafe_allow_html=True)
                render_obligations_table(obligations_list)


# ============================================================================
# TAB 3 — Gap Analysis (with charts)
# ============================================================================

with tab3:
    st.markdown('<p class="ai-eyebrow">Mode 3</p><h3>See where your current implementation falls short</h3>', unsafe_allow_html=True)
    sys_desc_3 = st.text_area("Describe your AI system", placeholder="An AI tool that screens job applicants...", height=80, key="gap_sys")

    st.markdown("**Your current implementation and controls**")
    input_source = st.radio("Input source", ["Type description", "Upload PDF document"], horizontal=True, key="gap_input_source", label_visibility="collapsed")

    current_state_3 = ""
    if input_source == "Type description":
        current_state_3 = st.text_area(
            "Current state",
            placeholder=("We use AI scores as one of several inputs to HR decisions. A human reviews top candidates. We log model outputs. We have not documented our risk management process..."),
            height=140, key="gap_state_text", label_visibility="collapsed",
        )
    else:
        uploaded_pdf = st.file_uploader("Upload a PDF describing your current implementation", type=["pdf"], key="gap_pdf")
        if uploaded_pdf is not None:
            with st.spinner("Extracting text..."):
                current_state_3 = extract_pdf_text(uploaded_pdf)
            if current_state_3.startswith("[error"):
                st.error(current_state_3)
                current_state_3 = ""
            else:
                st.success(f"Extracted {len(current_state_3)} characters from {uploaded_pdf.name}")
                with st.expander("Preview extracted text"):
                    preview = current_state_3[:2000]
                    if len(current_state_3) > 2000:
                        preview += "..."
                    st.text(preview)

    col_risk, col_btn = st.columns([2, 1])
    with col_risk:
        risk_3 = st.selectbox("Risk category", ["Auto classify first", "HIGH", "LIMITED", "MINIMAL", "PROHIBITED"], key="gap_risk")
    risk_3_value = None if risk_3 == "Auto classify first" else risk_3
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        run_gap = st.button("Run audit", type="primary", key="gap_btn", use_container_width=True)

    if run_gap and sys_desc_3.strip() and current_state_3.strip():
        with st.spinner("Auditing compliance against the EU AI Act..."):
            result = cached_gap_analysis(sys_desc_3, current_state_3, risk_3_value)
        st.session_state["gap_result"] = result
        st.session_state["gap_sys_desc"] = sys_desc_3
        st.session_state["gap_state"] = current_state_3

    if "gap_result" in st.session_state:
        result = st.session_state["gap_result"]
        if "error" in result:
            st.error(f"Could not run audit. {result['error']}")
        else:
            summary = result.get("summary", {})
            score = summary.get("compliance_score_percent", 0)
            risk = result.get("risk_category", "?")
            gap_items = result.get("gap_analysis", [])
            critical_gaps_count = summary.get("critical_gaps_count", 0)
            met = summary.get("met", 0)
            partial = summary.get("partial", 0)
            gaps = summary.get("gaps", 0)

            risk_label = {"PROHIBITED": "Prohibited", "HIGH": "High risk", "LIMITED": "Limited risk", "MINIMAL": "Minimal risk"}.get(risk, risk)
            interpretation = (
                f"Your system is classified as <b>{risk_label}</b>. Out of {summary.get('total_requirements', '?')} "
                f"EU AI Act obligations, you fully meet <b>{met}</b>, partially meet <b>{partial}</b>, "
                f"and have not started on <b>{gaps}</b>. <b>{critical_gaps_count} of those are critical</b> and need attention before deployment in the EU."
            )

            st.markdown('<div class="ai-section-header">Compliance status</div>', unsafe_allow_html=True)

            # Hero: gauge + interpretation
            col_score, col_summary = st.columns([1, 2])
            with col_score:
                st.plotly_chart(make_compliance_gauge(score), use_container_width=True, config={"displayModeBar": False})
            with col_summary:
                st.markdown(
                    f'<div style="padding: 16px 18px; border:1px solid var(--ai-border); border-radius:8px; background: var(--ai-bg-secondary); height: 160px; display: flex; align-items: center;">'
                    f'<div style="font-size:14px; line-height:1.6;">{interpretation}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Met", met)
            m2.metric("Partial", partial)
            m3.metric("Gap", gaps)
            m4.metric("Critical", critical_gaps_count)

            # ----- Charts row -----
            st.markdown('<div class="ai-section-header">Visual breakdown</div>', unsafe_allow_html=True)
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.markdown('<div class="ai-chart-title">Status breakdown</div>', unsafe_allow_html=True)
                st.plotly_chart(make_status_donut(met, partial, gaps), use_container_width=True, config={"displayModeBar": False})
            with chart_col2:
                st.markdown('<div class="ai-chart-title">Priority distribution</div>', unsafe_allow_html=True)
                st.plotly_chart(make_priority_bars(gap_items), use_container_width=True, config={"displayModeBar": False})

            # ----- Start here this week -----
            critical_items = [it for it in gap_items if it.get("priority") == "CRITICAL" and it.get("status") in ("GAP", "PARTIAL")][:3]
            if critical_items:
                st.markdown('<div class="ai-section-header">Start here this week</div>', unsafe_allow_html=True)
                st.caption(f"Top {len(critical_items)} highest impact actions, ranked by priority and impact.")
                for i, item in enumerate(critical_items, start=1):
                    title = item.get("requirement_title", "?")
                    article = item.get("article", "?")
                    action = item.get("recommended_action", "")
                    st.markdown(
                        f'<div class="ai-action">'
                        f'<div class="ai-action-num">{i}</div>'
                        f'<div class="ai-action-content">'
                        f'<div class="ai-action-title">{title}</div>'
                        f'<div class="ai-action-meta">{article}</div>'
                        f'<div class="ai-action-desc">{action}</div>'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )

            # ----- Full findings table -----
            st.markdown('<div class="ai-section-header">Full audit</div>', unsafe_allow_html=True)
            findings_df = format_findings_dataframe(gap_items)

            col_filter, col_dl_pdf, col_dl_md = st.columns([2, 1, 1])
            with col_filter:
                show_filter = st.radio("Show", ["All", "Critical only", "Gaps only", "Partial only"],
                                       horizontal=True, key="gap_filter", label_visibility="collapsed")
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            with col_dl_pdf:
                try:
                    pdf_bytes = generate_gap_report_pdf(result, st.session_state.get("gap_sys_desc", ""), st.session_state.get("gap_state", ""))
                    st.download_button(label="Download PDF", data=pdf_bytes,
                                       file_name=f"aiactchecker_gap_analysis_{stamp}.pdf",
                                       mime="application/pdf", key="gap_download_pdf",
                                       use_container_width=True, type="primary")
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")
            with col_dl_md:
                report_md = generate_gap_report_markdown(result, st.session_state.get("gap_sys_desc", ""), st.session_state.get("gap_state", ""))
                st.download_button(label="Download MD", data=report_md,
                                   file_name=f"aiactchecker_gap_analysis_{stamp}.md",
                                   mime="text/markdown", key="gap_download_md", use_container_width=True)

            filtered_df = findings_df.copy()
            if show_filter == "Critical only":
                filtered_df = filtered_df[filtered_df["Priority"] == "Critical"]
            elif show_filter == "Gaps only":
                filtered_df = filtered_df[filtered_df["Status"] == "Gap"]
            elif show_filter == "Partial only":
                filtered_df = filtered_df[filtered_df["Status"] == "Partial"]
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)


# ============================================================================
# TAB 4 — Cross Standard Map (with reference sanitizer)
# ============================================================================

with tab4:
    st.markdown('<p class="ai-eyebrow">Mode 4</p><h3>See how a governance topic is treated across three frameworks</h3>', unsafe_allow_html=True)
    topic_4 = st.text_input("Topic to map", placeholder="Data governance and data quality for AI systems", key="cross_topic")
    col_btn = st.columns([1, 4])[0]
    with col_btn:
        run_cross = st.button("Map topic", type="primary", key="cross_btn", use_container_width=True)

    if run_cross and topic_4.strip():
        with st.spinner("Mapping topic across EU AI Act, GDPR, and NIST AI RMF..."):
            result = cached_cross_map(topic_4)
        st.session_state["cross_result"] = result

    if "cross_result" in st.session_state:
        result = st.session_state["cross_result"]
        if "error" in result:
            st.error(f"Could not map topic. {result['error']}")
        else:
            mappings = result.get("mappings", {})
            st.markdown(f'<div class="ai-eyebrow" style="margin-top:24px;">Topic mapped</div><h3>{result.get("topic", "?")}</h3>', unsafe_allow_html=True)
            st.markdown('<div class="ai-section-header">How each framework addresses this</div>', unsafe_allow_html=True)
            col_eu, col_gd, col_ni = st.columns(3)

            def render_framework_card(col, mapping: dict, label: str, kind: str):
                with col:
                    raw_refs = mapping.get("primary_references", [])
                    refs = sanitize_references(raw_refs)
                    summary_text = mapping.get("summary", "")
                    obligations = mapping.get("key_obligations", [])
                    refs_str = ", ".join(refs[:4]) if refs else "Article references not available"
                    binding_label = "Binding law" if kind == "binding" else "Voluntary"
                    binding_class = "binding" if kind == "binding" else "voluntary"
                    obligations_html = ""
                    if obligations:
                        items = "".join([f"<li>{ob}</li>" for ob in obligations[:4]])
                        obligations_html = (
                            f'<div class="ai-framework-section">'
                            f'<div class="ai-framework-label">Key obligations</div>'
                            f'<ul style="font-size:12px; margin: 4px 0 0; padding-left: 16px; color: var(--ai-text-secondary); line-height: 1.5;">{items}</ul>'
                            f'</div>'
                        )
                    st.markdown(
                        f'<div class="ai-framework-card">'
                        f'<span class="ai-badge {binding_class}">{binding_label}</span>'
                        f'<div class="ai-framework-name">{label}</div>'
                        f'<div class="ai-framework-section"><div class="ai-framework-label">Anchors</div><div class="ai-framework-content" style="font-weight:500;">{refs_str}</div></div>'
                        f'<div class="ai-framework-section"><div class="ai-framework-label">Summary</div><div class="ai-framework-content">{summary_text or "No summary available."}</div></div>'
                        f'{obligations_html}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            render_framework_card(col_eu, mappings.get("eu_ai_act", {}), "EU AI Act", "binding")
            render_framework_card(col_gd, mappings.get("gdpr", {}), "GDPR", "binding")
            render_framework_card(col_ni, mappings.get("nist_ai_rmf", {}), "NIST AI RMF", "voluntary")

            overlap = result.get("overlap", "")
            differences = result.get("differences", "")
            guidance = result.get("compliance_guidance", "")
            if guidance:
                st.markdown(f'<div class="ai-callout" style="margin-top:24px;"><div class="ai-callout-title">Compliance shortcut</div><div class="ai-callout-body">{guidance}</div></div>', unsafe_allow_html=True)
            if differences:
                st.markdown(f'<div class="ai-callout"><div class="ai-callout-title">Where they differ</div><div class="ai-callout-body">{differences}</div></div>', unsafe_allow_html=True)
            if overlap:
                with st.expander("See overlap details"):
                    st.write(overlap)
