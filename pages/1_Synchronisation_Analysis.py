"""Page 1 — Overall & per-strategy synchronisation scores with heatmap."""

import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Synchronisation Analysis", page_icon="📊", layout="wide")
st.title("📊 Synchronisation Analysis")

# ── load analyser ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading analyser …")
def load():
    from src.analyzer import SyncAnalyzer
    return SyncAnalyzer()

az = load()

# ── overall score ───────────────────────────────────────────────────
overall = az.overall_score()
st.metric("Overall Synchronisation Score", f"{overall:.1f} %")
if overall >= 60:
    st.success("Good overall alignment between the strategic plan and action plan.")
elif overall >= 40:
    st.warning("Moderate alignment — several strategic areas need stronger action coverage.")
else:
    st.error("Weak alignment — the action plan has significant gaps relative to the strategy.")
st.divider()

# ── per-strategy scores ────────────────────────────────────────────
st.subheader("Per-Strategy Alignment Scores")
scores = az.strategy_scores()
cols = st.columns(len(scores))
for i, (k, v) in enumerate(scores.items()):
    with cols[i]:
        st.metric(k, f"{v['top3_avg']:.1f} %", delta=f"best {v['best_match']:.1f} %")

st.divider()

# ── radar chart ─────────────────────────────────────────────────────
st.subheader("Radar — Strategy Alignment Profile")
aim_ids = [k for k in scores if not k.startswith("FC")]
aim_vals = [scores[k]["top3_avg"] for k in aim_ids]
fig_radar = go.Figure(go.Scatterpolar(
    r=aim_vals + [aim_vals[0]],
    theta=aim_ids + [aim_ids[0]],
    fill="toself", fillcolor="rgba(30,136,229,0.25)",
    line=dict(color="#1E88E5"),
))
fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                        template="plotly_white", height=400)
st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# ── heatmap ─────────────────────────────────────────────────────────
st.subheader("Alignment Heatmap — Strategic Items vs Action Items")
sim = az.similarity_matrix
s_labels = [t["id"] for t in az.strategic_texts]
a_labels = [t["id"] for t in az.action_texts]

fig_heat = px.imshow(
    sim, x=a_labels, y=s_labels,
    color_continuous_scale="Blues",
    labels=dict(x="Action Item", y="Strategic Item", color="Cosine Similarity"),
    aspect="auto",
    text_auto=".2f",  # Show numerical values with 2 decimal places
)
fig_heat.update_layout(height=500, template="plotly_white")
fig_heat.update_traces(textfont_size=8)  # Make text readable
st.plotly_chart(fig_heat, use_container_width=True)

# ── gap analysis ────────────────────────────────────────────────────
st.divider()
st.subheader("Gap Analysis")
threshold = st.slider("Similarity threshold", 0.2, 0.7, 0.45, 0.05)
gap = az.gap_analysis(threshold)

c1, c2 = st.columns(2)
with c1:
    st.markdown("#### ⚠️ Gaps (below threshold)")
    if gap["gaps"]:
        for g in gap["gaps"]:
            st.warning(f"**{g['id']}** — best similarity {g['best_sim']:.2f}")
    else:
        st.success("No gaps detected at this threshold.")
with c2:
    st.markdown("#### ✅ Strengths (above threshold)")
    for s in gap["strengths"]:
        st.info(f"**{s['id']}** — best similarity {s['best_sim']:.2f}")
