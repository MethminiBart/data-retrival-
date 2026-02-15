"""
ISPS — Intelligent Strategic Plan Synchronisation System
Home page: overview dashboard for HHS Vision 2030 ↔ QIP 2025-26.
"""

import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # loads .env file automatically

st.set_page_config(
    page_title="ISPS — Strategic Plan Sync",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hospital-3.png", width=64)
    st.markdown("### ISPS Dashboard")
    st.caption("Hamilton Health Sciences\nVision 2030 ↔ QIP 2025-26")
    st.divider()
    env_key = os.getenv("GEMINI_API_KEY", "")
    api_key = st.text_input("Google Gemini API Key (optional)", value=env_key, type="password",
                            help="Loaded from .env automatically. Or paste manually.")
    if api_key:
        st.session_state["gemini_key"] = api_key
        st.success("API key set ✓")
    st.divider()
    st.markdown(
        "**Pages**\n"
        "1. 📊 Synchronisation Analysis\n"
        "2. 🔍 Strategy Deep Dive\n"
        "3. 💡 Improvement Suggestions\n"
        "4. 🕸️ Knowledge Graph\n"
        "5. 📈 Evaluation Metrics\n"
        "6. 💬 Chat With Plans"
    )

# ── cached analyser ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model & computing scores …")
def load_analyzer():
    from src.analyzer import SyncAnalyzer
    return SyncAnalyzer()

analyzer = load_analyzer()

# ── hero section ────────────────────────────────────────────────────
st.title("🏥 Intelligent Strategic Plan Synchronisation System")
st.markdown(
    "Analyse how well **Hamilton Health Sciences' QIP 2025-26** (Action Plan) "
    "aligns with the **Vision 2030** Strategic Plan using NLP embeddings, "
    "vector databases, RAG and knowledge-graph visualisation."
)
st.divider()

# ── key metrics row ─────────────────────────────────────────────────
overall = analyzer.overall_score()
scores = analyzer.strategy_scores()
gaps = analyzer.gap_analysis()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Sync Score", f"{overall:.1f} %")
col2.metric("Strategic Aims Analysed", len([s for s in scores if not s.startswith("FC")]))
col3.metric("Action Items Mapped", len(analyzer.action_texts))
col4.metric("Gaps Detected", len(gaps["gaps"]))

st.divider()

# ── quick bar chart ─────────────────────────────────────────────────
import plotly.graph_objects as go

labels = list(scores.keys())
vals = [scores[k]["top3_avg"] for k in labels]
colours = ["#1E88E5" if not k.startswith("FC") else "#7B1FA2" for k in labels]

fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=colours,
                       text=[f"{v:.1f} %" for v in vals], textposition="auto"))
fig.update_layout(title="Alignment Score by Strategic Item (top-3 avg)",
                  yaxis_title="Score (%)", xaxis_title="Strategic Item",
                  template="plotly_white", height=380)
st.plotly_chart(fig, use_container_width=True)

# ── documents overview ──────────────────────────────────────────────
c1, c2 = st.columns(2)
with c1:
    st.subheader("📄 Strategic Plan — Vision 2030")
    st.markdown(
        "**Vision:** " + "To shape the future of health by leading in care, "
        "discovery, and learning, while advancing equity and regional growth.\n\n"
        "**Strategic Aims:** LEAD · BUILD · SERVE · LEARN\n\n"
        "**Values:** Collaboration · Compassion · Advancement · Accountability"
    )
with c2:
    st.subheader("📋 Action Plan — QIP 2025-26")
    st.markdown(
        "**7 QIP Priorities** including sepsis reduction, surgical safety, "
        "pressure injuries, hand hygiene, EDI data, documentation and equity "
        "in critical incidents.\n\n"
        "**8 Action Areas:** Access & Flow · Equity & Indigenous Health · "
        "Patient Experience · Provider Experience · Safety · Palliative Care · "
        "Population Health · ED Return Visits"
    )

st.divider()
st.info("👈 Use the sidebar to navigate to detailed analysis pages.")
