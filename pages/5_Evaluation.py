"""Page 5 — Evaluation metrics: precision, recall, F1, confusion matrix."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

st.set_page_config(page_title="Evaluation Metrics", page_icon="📈", layout="wide")
st.title("📈 Evaluation & Testing")
st.markdown(
    "Compare the system's **embedding-based alignment detection** against a "
    "**manually defined ground truth** to measure precision, recall, and F1."
)

# ── analyser ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading analyser …")
def load():
    from src.analyzer import SyncAnalyzer
    return SyncAnalyzer()

az = load()

from src.evaluation import evaluate, evaluate_thresholds

# ── threshold selector ──────────────────────────────────────────────
threshold = st.slider("Similarity threshold for alignment detection", 0.20, 0.70, 0.45, 0.05)
metrics = evaluate(az.similarity_matrix, threshold)

# ── overall metrics ─────────────────────────────────────────────────
st.subheader("Overall Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision", f"{metrics['overall']['precision']:.2f}")
c2.metric("Recall", f"{metrics['overall']['recall']:.2f}")
c3.metric("F1 Score", f"{metrics['overall']['f1']:.2f}")
c4.metric("Threshold", f"{threshold:.2f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("True Positives", metrics["true_pos"])
c6.metric("False Positives", metrics["false_pos"])
c7.metric("False Negatives", metrics["false_neg"])
c8.metric("True Negatives", metrics["true_neg"])

st.divider()

# ── confusion matrix ────────────────────────────────────────────────
st.subheader("Confusion Matrix")
cm = np.array(metrics["confusion_matrix"])
fig_cm = px.imshow(
    cm, text_auto=True,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=["Not Aligned", "Aligned"],
    y=["Not Aligned", "Aligned"],
    color_continuous_scale="Blues",
)
fig_cm.update_layout(template="plotly_white", height=350, width=400)
st.plotly_chart(fig_cm)

st.divider()

# ── per-strategy metrics ───────────────────────────────────────────
st.subheader("Per-Strategy Metrics")
ps = metrics["per_strategy"]
df = pd.DataFrame([
    {"Strategy": k, "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1"]}
    for k, v in ps.items()
])
st.dataframe(df.style.format({"Precision": "{:.2f}", "Recall": "{:.2f}", "F1": "{:.2f}"}),
             use_container_width=True, hide_index=True)

fig_ps = px.bar(df.melt(id_vars="Strategy", var_name="Metric", value_name="Score"),
                x="Strategy", y="Score", color="Metric", barmode="group",
                template="plotly_white", height=400)
st.plotly_chart(fig_ps, use_container_width=True)

st.divider()

# ── threshold sweep ─────────────────────────────────────────────────
st.subheader("Threshold Optimisation")
sweep = evaluate_thresholds(az.similarity_matrix)
df_sweep = pd.DataFrame(sweep)

fig_sweep = go.Figure()
for metric in ["precision", "recall", "f1"]:
    fig_sweep.add_trace(go.Scatter(
        x=df_sweep["threshold"], y=df_sweep[metric],
        mode="lines+markers", name=metric.capitalize(),
    ))
fig_sweep.update_layout(
    xaxis_title="Threshold", yaxis_title="Score",
    template="plotly_white", height=400,
    title="Precision / Recall / F1 across Thresholds",
)
st.plotly_chart(fig_sweep, use_container_width=True)

best = max(sweep, key=lambda x: x["f1"])
st.success(f"Optimal threshold: **{best['threshold']:.2f}** "
           f"(F1 = {best['f1']:.2f}, P = {best['precision']:.2f}, R = {best['recall']:.2f})")
