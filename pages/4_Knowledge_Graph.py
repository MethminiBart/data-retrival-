"""Page 4 — Interactive knowledge-graph visualisation."""

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Knowledge Graph", page_icon="🕸️", layout="wide")
st.title("🕸️ Knowledge Graph")
st.markdown(
    "Interactive graph showing relationships between **strategic aims** (blue diamonds), "
    "**foundational commitments** (purple), **values** (teal), and **action items** (orange)."
)

from src.knowledge_graph import build_graph, render_pyvis, graph_stats

G = build_graph()

# ── stats row ───────────────────────────────────────────────────────
stats = graph_stats(G)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Nodes", stats["nodes"])
c2.metric("Total Edges", stats["edges"])
c3.metric("Strategic Aims", stats["strategic_aims"])
c4.metric("Action Items", stats["actions"])

st.divider()

# ── interactive graph ───────────────────────────────────────────────
html_path = render_pyvis(G)
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()
components.html(html, height=640, scrolling=True)

# ── legend ──────────────────────────────────────────────────────────
st.divider()
st.subheader("Legend")
st.markdown(
    "| Colour | Shape | Meaning |\n"
    "|--------|-------|---------|\n"
    "| 🔵 Blue | Diamond | Strategic Aim |\n"
    "| 🟣 Purple | Dot | Foundational Commitment |\n"
    "| 🟢 Teal | Dot | Value |\n"
    "| 🟠 Orange | Dot | Action Item |\n"
)

st.markdown(
    "**Edge types:** "
    "*supports* (action → strategy), "
    "*guides* (value → strategy), "
    "*underpins* (commitment → strategy)"
)

st.divider()
st.subheader("Graph Metrics")
st.json(stats)
