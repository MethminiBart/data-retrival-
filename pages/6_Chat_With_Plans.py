"""Page 6 — RAG-powered chat: ask questions about both plans."""

import streamlit as st

st.set_page_config(page_title="Chat With Plans", page_icon="💬", layout="wide")
st.title("💬 Chat With Plans")
st.markdown(
    "Ask any question about the **Vision 2030 Strategic Plan** or the "
    "**QIP 2025-26 Action Plan**. The system retrieves relevant document "
    "chunks from ChromaDB and sends them to Gemini for an answer."
)

# ── analyser ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading analyser …")
def load():
    from src.analyzer import SyncAnalyzer
    return SyncAnalyzer()

az = load()

# ── check API key ───────────────────────────────────────────────────
api_key = st.session_state.get("gemini_key", "")
if not api_key:
    api_key = st.text_input("Enter Google Gemini API key", type="password",
                            help="Free at https://aistudio.google.com/apikey")
    if api_key:
        st.session_state["gemini_key"] = api_key

if not api_key:
    st.warning("Enter a **free** Google Gemini API key to enable the chat.")
    st.stop()

from src.llm_helper import LLMHelper
llm = LLMHelper(api_key)

# ── chat state ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ── display history ─────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── input ───────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question about the strategic or action plan …"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG: retrieve relevant chunks from both collections
    action_hits = az.query(prompt, "action_plan", n=4)
    strat_hits = az.query(prompt, "strategic_plan", n=3)
    context_chunks = [h["text"] for h in strat_hits + action_hits]

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context & generating answer …"):
            answer = llm.chat(prompt, context_chunks, st.session_state["messages"])
        st.markdown(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})

# ── example questions ───────────────────────────────────────────────
with st.expander("Example questions you can ask"):
    st.markdown(
        "- How does the QIP support the SERVE strategic aim?\n"
        "- What actions address health equity in the QIP?\n"
        "- Are there any gaps in the BUILD strategy's action coverage?\n"
        "- How is Indigenous health addressed in the action plan?\n"
        "- What safety initiatives are in the QIP?\n"
        "- How does palliative care align with the strategic vision?\n"
        "- What is the Care Transformation initiative?"
    )
