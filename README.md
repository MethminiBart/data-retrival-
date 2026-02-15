# ISPS — Intelligent Strategic Plan Synchronisation System

A smart AI-based system that analyses the synchronisation between
**Hamilton Health Sciences' Vision 2030** (Strategic Plan) and their
**Quality Improvement Plan 2025-26** (Action Plan).

Built for the MSc Data Science — Information Retrieval coursework.

---

## Features

| Feature | Technology |
|---------|-----------|
| Embedding-based synchronisation scoring | sentence-transformers + cosine similarity |
| Vector database storage & retrieval | ChromaDB |
| Interactive dashboard with charts | Streamlit + Plotly |
| RAG-powered improvement suggestions | Google Gemini + ChromaDB |
| Agentic AI multi-step reasoning | Google Gemini |
| Knowledge graph visualisation | NetworkX + PyVis |
| Precision / Recall / F1 evaluation | scikit-learn |
| Interactive chat with plans | RAG (ChromaDB → Gemini) |

## Quick Start

```bash
# 1 — Install dependencies
pip install -r requirements.txt

# 2 — Run the dashboard
streamlit run app.py
```

The core analysis (sync scores, heatmaps, knowledge graph, evaluation)
works **without any API key**.

For AI-powered features (suggestions, chat, agentic analysis), enter a
**free** Google Gemini API key in the sidebar:
→ https://aistudio.google.com/apikey

## Project Structure

```
├── app.py                        # Home page
├── pages/
│   ├── 1_Synchronisation_Analysis.py
│   ├── 2_Strategy_Deep_Dive.py
│   ├── 3_Improvement_Suggestions.py
│   ├── 4_Knowledge_Graph.py
│   ├── 5_Evaluation.py
│   └── 6_Chat_With_Plans.py
├── src/
│   ├── document_processor.py     # PDF extraction & structured data
│   ├── analyzer.py               # Embeddings, ChromaDB, sync scores
│   ├── llm_helper.py             # Google Gemini integration
│   ├── knowledge_graph.py        # NetworkX graph builder
│   └── evaluation.py             # Precision / recall / F1
├── data/
│   ├── strategic_plan.pdf        # HHS Vision 2030
│   └── action_plan.pdf           # QIP 2025-26
├── requirements.txt
└── .streamlit/config.toml
```

## Architecture

```
Strategic Plan (PDF) ──┐
                       ├─→ Document Processor ─→ Structured Data
Action Plan (PDF) ─────┘          │
                                  ▼
                       Sentence-Transformers ─→ Embeddings
                                  │
                           ┌──────┴──────┐
                           ▼             ▼
                       ChromaDB     Cosine Similarity
                       (Vector DB)   Matrix
                           │             │
                   ┌───────┴───┐    ┌────┴────┐
                   ▼           ▼    ▼         ▼
                RAG Chat   Suggestions  Scores  Heatmap
                   │           │         │
                   └─────┬─────┘    ┌────┘
                         ▼          ▼
                    Google Gemini  Evaluation
                    (Free tier)   (P/R/F1)
                         │
                    ┌────┴─────┐
                    ▼          ▼
              Streamlit    Knowledge
              Dashboard    Graph (PyVis)
```

## Technologies Used

- **Streamlit** — interactive web dashboard
- **sentence-transformers** — local embedding model (all-MiniLM-L6-v2)
- **ChromaDB** — vector database for document retrieval
- **Google Gemini** — LLM for RAG, suggestions, agentic reasoning
- **NetworkX + PyVis** — knowledge graph construction & visualisation
- **Plotly** — charts (radar, heatmap, bar, scatter)
- **scikit-learn** — evaluation metrics (precision, recall, F1)
- **pypdf** — PDF text extraction

## Input Documents

1. **Strategic Plan**: Hamilton Health Sciences — Vision 2030
2. **Action Plan**: Hamilton Health Sciences — Quality Improvement Plan 2025-26
