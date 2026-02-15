"""Page 3 — RAG-powered improvement suggestions & agentic reasoning."""

import streamlit as st

st.set_page_config(page_title="Improvement Suggestions", page_icon="💡", layout="wide")
st.title("💡 Improvement Suggestions")

# ── analyser ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading analyser …")
def load():
    from src.analyzer import SyncAnalyzer
    return SyncAnalyzer()

az = load()

# ── check API key ───────────────────────────────────────────────────
api_key = st.session_state.get("gemini_key", "")
if not api_key:
    api_key = st.text_input("Enter your Google Gemini API key to enable AI suggestions",
                            type="password",
                            help="Free at https://aistudio.google.com/apikey")
    if api_key:
        st.session_state["gemini_key"] = api_key

if not api_key:
    st.warning("Enter a **free** Google Gemini API key in the sidebar (or above) to unlock AI-powered suggestions.")
    st.stop()

from src.llm_helper import LLMHelper
llm = LLMHelper(api_key)

from src.document_processor import STRATEGIC_OBJECTIVES, ACTION_ITEMS

# ── tabs: per-strategy suggestions & executive report & agentic ────
tab1, tab2, tab3 = st.tabs(["Per-Strategy Suggestions", "Executive Report", "Agentic Analysis"])

scores = az.strategy_scores()
details = az.alignment_details()
action_lookup = {a.id: a for a in ACTION_ITEMS}

# ── Tab 1: per-strategy ────────────────────────────────────────────
with tab1:
    choice = st.selectbox("Select a strategic item", [s.code for s in STRATEGIC_OBJECTIVES])
    obj = next(o for o in STRATEGIC_OBJECTIVES if o.code == choice)
    detail = next(d for d in details if d["strategic_id"] == choice)

    top_names = [
        action_lookup[a["action_id"]].title
        for a in detail["top_actions"]
        if a["action_id"] in action_lookup
    ]

    if st.button("Generate AI Suggestions", key="suggest"):
        with st.spinner("Gemini is thinking …"):
            result = llm.suggest_improvements(
                f"{obj.name}: {obj.description}",
                top_names,
                detail["score"],
            )
        st.markdown(result)

# ── Tab 2: executive report ─────────────────────────────────────────
with tab2:
    if st.button("Generate Executive Report", key="exec"):
        with st.spinner("Generating executive summary …"):
            gaps = az.gap_analysis()
            report = llm.executive_report(
                az.overall_score(), scores, gaps["gaps"]
            )
        st.markdown(report)

# ── Tab 3: agentic analysis ─────────────────────────────────────────
with tab3:
    st.markdown(
        "The **Agentic AI** autonomously reasons through a multi-step "
        "analysis chain to explore alignment gaps in depth."
    )
    agent_choice = st.selectbox("Strategic aim for deep analysis",
                                [s.code for s in STRATEGIC_OBJECTIVES],
                                key="agent_sel")
    agent_obj = next(o for o in STRATEGIC_OBJECTIVES if o.code == agent_choice)

    all_action_names = [a.title for a in ACTION_ITEMS]

    if st.button("Run Agentic Analysis", key="agent"):
        with st.spinner("Agent reasoning (this may take a moment) …"):
            result = llm.agentic_analysis(
                f"{agent_obj.name}: {agent_obj.description}",
                all_action_names,
                agent_obj.focus_areas[0],
            )
        st.markdown(result)
