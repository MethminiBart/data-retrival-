"""Page 2 — Drill into each strategic objective and its aligned actions."""

import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Strategy Deep Dive", page_icon="🔍", layout="wide")
st.title("🔍 Strategy Deep Dive")

@st.cache_resource(show_spinner="Loading analyser …")
def load():
    from src.analyzer import SyncAnalyzer
    return SyncAnalyzer()

az = load()
details = az.alignment_details()

from src.document_processor import STRATEGIC_OBJECTIVES, FOUNDATIONAL_COMMITMENTS, ACTION_ITEMS

action_lookup = {a.id: a for a in ACTION_ITEMS}

# ── selector ────────────────────────────────────────────────────────
names = {d["strategic_id"]: d["strategic_id"] for d in details}
choice = st.selectbox("Select a strategic item to explore", list(names.keys()))

item = next(d for d in details if d["strategic_id"] == choice)

st.metric(f"Alignment Score — {choice}", f"{item['score']:.1f} %")
st.divider()

# ── top aligned actions ─────────────────────────────────────────────
st.subheader("Top 5 Aligned Actions")
for rank, a in enumerate(item["top_actions"], 1):
    aid = a["action_id"]
    label = action_lookup[aid].title if aid in action_lookup else aid
    pct = a["similarity"] * 100
    colour = "green" if pct >= 50 else "orange" if pct >= 35 else "red"
    st.markdown(f"**{rank}. {aid}** — {label}")
    st.progress(min(a["similarity"], 1.0))
    st.caption(f"Cosine similarity: {a['similarity']:.3f} ({pct:.1f} %)")

st.divider()

# ── weakest actions ─────────────────────────────────────────────────
st.subheader("Weakest Aligned Actions")
for a in item["weak_actions"]:
    aid = a["action_id"]
    label = action_lookup[aid].title if aid in action_lookup else aid
    st.markdown(f"- **{aid}** — {label} (similarity {a['similarity']:.3f})")

st.divider()

# ── bar chart ───────────────────────────────────────────────────────
st.subheader(f"All Action Similarities for {choice}")
all_scores = sorted(
    [
        {"Action": t["action_id"],
         "Section": t["section"],
         "Similarity": t["similarity"]}
        for t in item["top_actions"]
    ] + [
        {"Action": t["action_id"],
         "Section": t["section"],
         "Similarity": t["similarity"]}
        for t in item["weak_actions"]
    ],
    key=lambda x: x["Similarity"], reverse=True,
)

# Actually get all actions for this strategy
sim_row = az.similarity_matrix[[i for i, s in enumerate(az.strategic_texts) if s["id"] == choice][0]]
df = pd.DataFrame({
    "Action": [t["id"] for t in az.action_texts],
    "Section": [t["section"] for t in az.action_texts],
    "Similarity": sim_row,
})
df = df.sort_values("Similarity", ascending=True)

fig = px.bar(df, x="Similarity", y="Action", color="Section", orientation="h",
             template="plotly_white", height=600)
fig.update_layout(xaxis_title="Cosine Similarity", yaxis_title="")
st.plotly_chart(fig, use_container_width=True)
