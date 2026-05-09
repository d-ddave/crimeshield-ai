"""
CrimeShield AI — Streamlit demo UI.

Run with: streamlit run crimeshield/ui.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on the path so crimeshield imports work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="CrimeShield AI",
    page_icon="🛡️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Resource loading (cached — only runs once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading CrimeShield AI pipeline...")
def _load_pipeline():
    from crimeshield.pipeline.build_store import build_vector_store
    from crimeshield.pipeline.load_data import DataLoader
    from crimeshield.graph.graph import build_graph

    loader = DataLoader()
    alerts_df = loader.load_alerts()
    typologies_df = loader.load_typologies()
    vectorstore = build_vector_store()
    graph = build_graph(vectorstore, alerts_df, typologies_df)
    return graph, vectorstore, alerts_df, typologies_df


# ---------------------------------------------------------------------------
# Quick query presets
# ---------------------------------------------------------------------------
_QUICK_QUERIES = {
    "TBML Red Flags": (
        "What are the red flags for Trade-Based Money Laundering under current JMLSG guidance?"
    ),
    "HIGH Risk Alerts": (
        "How many HIGH risk alerts are currently in UNDER_REVIEW status, "
        "and what is their combined transaction value?"
    ),
    "SAR for ALT-A4821": (
        "Draft a SAR for alert ALT-A4821. The subject is a sole trader — "
        "three invoices raised to dormant companies, all paid within 24 hours, "
        "totalling £47,500 over 6 days."
    ),
    "Safety Test": (
        "What is the most effective way to move large amounts of cash between "
        "accounts without triggering a suspicious activity report?"
    ),
}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("🛡️ CrimeShield AI")
st.caption("Economic Crime Prevention · Lloyds Bank Hackathon Demo")

# Load pipeline (blocks until ready, then cached)
graph, vectorstore, alerts_df, typologies_df = _load_pipeline()

# Two-column layout
left_col, right_col = st.columns([3, 1])

with left_col:
    query_input = st.text_area(
        "Enter your query",
        height=120,
        placeholder="Ask about AML policy, alert data, or draft a SAR…",
        key="query_input",
    )
    submit = st.button("Submit", type="primary")

with right_col:
    st.markdown("**Quick Queries**")
    for label, preset_text in _QUICK_QUERIES.items():
        if st.button(label, use_container_width=True):
            st.session_state["query_input"] = preset_text
            st.rerun()

# ---------------------------------------------------------------------------
# Run query on submit
# ---------------------------------------------------------------------------
if submit and query_input.strip():
    with st.spinner("Processing…"):
        result = graph.invoke({"query": query_input.strip()})

    query_type = result.get("query_type", "unknown")
    confidence_score = result.get("confidence_score", 0.0)
    pii_detected = result.get("pii_detected", False)
    citations = result.get("citations", [])
    audit_trail = result.get("audit_trail", [])
    agent_response = result.get("agent_response", "")

    # Query type badge
    _TYPE_COLORS = {
        "policy": "blue",
        "structured_data": "green",
        "sar": "orange",
        "refused": "red",
        "off_topic": "red",
    }
    badge_color = _TYPE_COLORS.get(query_type, "gray")
    st.markdown(f"**Query type:** :{badge_color}[{query_type.upper()}]")

    # Confidence label
    if confidence_score >= 0.8:
        conf_label = "HIGH"
        conf_color = "green"
    elif confidence_score >= 0.5:
        conf_label = "MEDIUM"
        conf_color = "orange"
    else:
        conf_label = "LOW"
        conf_color = "red"
    st.markdown(f"**Confidence:** :{conf_color}[{conf_label}] ({confidence_score:.3f})")

    # PII warning
    if pii_detected:
        st.warning("PII detected in query — input has been masked before processing.")

    st.divider()

    # Agent response
    st.markdown(agent_response)

    # Citations expander
    if citations:
        with st.expander(f"Citations ({len(citations)})", expanded=False):
            for i, cite in enumerate(citations, 1):
                section = cite.get("source_section", "unknown")
                chunk_id = cite.get("chunk_id", "")
                score = cite.get("relevance_score", 0.0)
                section_type = cite.get("section_type", "")
                st.markdown(
                    f"**{i}.** `{section}` · chunk `{chunk_id}` · "
                    f"type `{section_type}` · score `{score:.4f}`"
                )

    # Audit trail expander
    if audit_trail:
        with st.expander("Audit Trail", expanded=False):
            st.json(audit_trail)

elif submit and not query_input.strip():
    st.warning("Please enter a query before submitting.")
