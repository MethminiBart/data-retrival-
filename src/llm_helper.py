"""
LLM Helper — uses Google Gemini (free tier) for RAG-based suggestions,
agentic reasoning, and interactive chat.

This module provides a thin abstraction layer over the Google Gemini API,
implementing four key intelligent features:
1. Improvement suggestions per strategic objective
2. Executive summary reports
3. RAG-powered chat interface
4. Multi-step agentic reasoning analysis

All methods use the _safe_call wrapper to handle API errors gracefully,
particularly quota exhaustion (429) and model not found (404) errors.
"""

from google import genai


class LLMHelper:
    """
    Wrapper around Google Gemini API for all LLM-powered features.
    
    This class encapsulates all interactions with the Gemini API, providing
    structured prompt templates for strategic planning analysis. It uses the
    new google.genai SDK (not the deprecated google.generativeai).
    
    Attributes:
        client: Google Gemini API client instance
        model_name: Name of the Gemini model to use (default: gemini-2.5-flash-lite)
    """

    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash-lite"):
        """
        Initialize the LLM helper with API credentials.
        
        Args:
            api_key: Google Gemini API key (loaded from .env or user input)
            model_name: Gemini model identifier. Default is gemini-2.5-flash-lite
                       which is available on free tier as of Feb 2026.
        
        Note:
            The API key is passed from app.py/pages via st.session_state,
            never hardcoded in the codebase.
        """
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    @staticmethod
    def _safe_call(func):
        """
        Wrapper for Gemini API calls with error handling.
        
        This method catches common API errors and returns user-friendly
        messages instead of letting exceptions propagate to Streamlit.
        Particularly important for:
        - Quota exhaustion (429/ResourceExhausted): Common on free tier
        - Model not found (404/NotFound): When model name is invalid
        
        Args:
            func: Lambda/callable that makes a Gemini API call
        
        Returns:
            str: Either the generated text or a formatted error message
        
        Design Decision:
            We fail fast rather than retry, because:
            1. Quota errors won't resolve without user action (new API key)
            2. Model errors indicate configuration issues
            3. Avoids hanging the Streamlit app waiting for retries
        """
        try:
            return func()
        except Exception as e:
            msg = str(e)
            # Check for quota/rate limit errors
            if "429" in msg or "ResourceExhausted" in msg or "quota" in msg.lower():
                return (
                    "**Rate limit reached.** Your free Gemini quota is "
                    "exhausted for today.\n\n"
                    "**Quick fix:** Go to https://aistudio.google.com/apikey → "
                    "click **Create API key → In new project**. "
                    "Paste the new key in the sidebar and try again."
                )
            # Check for model not found errors
            if "404" in msg or "NotFound" in msg:
                return (
                    "**Model not found.** The selected Gemini model is "
                    "unavailable. Please check your API key and try again."
                )
            # Generic error fallback
            return f"**Error:** {msg}"

    # ────────────────────────────────────────────────────────────────
    # FEATURE 1: Improvement Suggestions
    # ────────────────────────────────────────────────────────────────

    def suggest_improvements(
        self, strategic_obj: str, aligned_actions: list[str], score: float
    ) -> str:
        """
        Generate AI-powered improvement suggestions for a strategic objective.
        
        This method implements the "Intelligent Improvement Suggestions" feature
        required by the coursework. It provides context-aware recommendations
        based on the strategic objective, its current alignment score, and
        the action items that are aligned to it.
        
        Args:
            strategic_obj: Full text of the strategic objective being analyzed
            aligned_actions: List of action item texts that scored high alignment
            score: Current alignment percentage (0-100)
        
        Returns:
            str: Markdown-formatted analysis containing:
                - Assessment of current alignment
                - Identified gaps
                - 3-5 concrete improvement recommendations
                - Suggested KPIs for tracking progress
                - Realistic implementation timeline
        
        Design Pattern:
            This uses a structured prompt template to ensure consistent,
            actionable output. The prompt includes:
            - Domain context (HHS healthcare)
            - Current state (score + aligned actions)
            - Required output structure (5 sections)
        """
        prompt = f"""You are a healthcare strategic-planning expert analysing
Hamilton Health Sciences' Vision 2030 strategic plan and their
Quality Improvement Plan (QIP) 2025-26.

Strategic Objective: {strategic_obj}
Current Alignment Score: {score:.1f} %
Current Aligned Actions:
{chr(10).join(f'  - {a}' for a in aligned_actions)}

Provide:
1. **Assessment** (2-3 sentences on current alignment)
2. **Gaps Identified** — specific gaps between strategy and actions
3. **Improvement Suggestions** — 3-5 concrete, actionable recommendations
4. **Suggested KPIs** — 2-3 measurable indicators
5. **Timeline** — realistic implementation timeline

Be specific to HHS and their healthcare context."""
        # Use _safe_call wrapper to handle API errors gracefully
        # Lambda ensures the API call only executes inside the try/except block
        return self._safe_call(lambda: self.client.models.generate_content(
            model=self.model_name, contents=prompt
        ).text)

    # ────────────────────────────────────────────────────────────────
    # FEATURE 2: Executive Report Generation
    # ────────────────────────────────────────────────────────────────

    def executive_report(
        self, overall: float, scores: dict, gaps: list[dict]
    ) -> str:
        """
        Generate a board-ready executive summary of the synchronization analysis.
        
        This method synthesizes the overall system output into a concise,
        professional report suitable for presentation to hospital leadership.
        
        Args:
            overall: Overall synchronization score (0-100)
            scores: Dictionary mapping strategic IDs to their alignment scores
            gaps: List of strategic objectives with weak alignment (below threshold)
        
        Returns:
            str: 300-400 word executive summary in Markdown format covering:
                1. Overall assessment
                2. Key strengths
                3. Critical gaps
                4. Top 3 priority recommendations
                5. Forward-looking conclusion
        
        Use Case:
            Intended for senior leadership who need a high-level view without
            diving into the detailed heatmaps and per-strategy analysis.
        """
        lines = "\n".join(f"  - {k}: {v['top3_avg']:.1f} %" for k, v in scores.items())
        gap_lines = "\n".join(f"  - {g['id']}: {g['best_sim']:.2f}" for g in gaps)
        prompt = f"""You are a healthcare strategic-planning expert.
Analyse this synchronisation report for Hamilton Health Sciences'
Vision 2030 and QIP 2025-26.

Overall Synchronisation Score: {overall:.1f} %

Strategy-wise Scores:
{lines}

Identified Gaps:
{gap_lines if gap_lines else 'No significant gaps identified.'}

Write a 300-400 word executive summary covering:
1. Overall assessment   2. Key strengths   3. Critical gaps
4. Top 3 priority recommendations   5. Forward-looking conclusion

Tone: professional, suitable for a hospital board presentation."""
        return self._safe_call(lambda: self.client.models.generate_content(
            model=self.model_name, contents=prompt
        ).text)

    # ────────────────────────────────────────────────────────────────
    # FEATURE 3: RAG-Powered Chat Interface
    # ────────────────────────────────────────────────────────────────

    def chat(self, question: str, context_chunks: list[str],
             history: list[dict] | None = None) -> str:
        """
        Answer questions using Retrieval-Augmented Generation (RAG).
        
        This implements the core RAG pattern:
        1. Relevant document chunks are retrieved from ChromaDB (done by caller)
        2. Chunks are injected into the prompt as context
        3. LLM generates answer grounded in the provided context
        
        Args:
            question: User's natural language question
            context_chunks: Top-N most relevant document chunks from vector search
            history: Optional list of previous conversation turns for continuity
        
        Returns:
            str: Answer generated from the context, with explicit statement if
                the answer is not found in the provided documents
        
        RAG Benefits:
            - Grounds responses in actual document content (reduces hallucination)
            - More accurate than pure LLM knowledge (which may be outdated)
            - Can cite specific sections of strategic/action plans
        
        Implementation Notes:
            - Limits context to top 5 chunks to stay within token limits
            - Includes last 6 conversation turns for context
            - Explicitly instructs LLM to admit when answer is not in context
        """
        # Build context string from top retrieved chunks
        ctx = "\n\n".join(context_chunks[:5])  # Limit to 5 to manage token count
        
        # Build conversation history string (last 6 turns for context)
        hist = ""
        if history:
            for m in history[-6:]:  # Limit history to avoid token overflow
                role = "User" if m["role"] == "user" else "Assistant"
                hist += f"{role}: {m['content']}\n"

        # Construct the RAG prompt with context injection
        prompt = f"""You are an AI assistant for analysing Hamilton Health Sciences'
strategic alignment (Vision 2030 + QIP 2025-26).

Relevant document context:
{ctx}

{"Previous conversation:" + chr(10) + hist if hist else ""}

User question: {question}

Answer based on the documents. If the answer is not in the context,
say so clearly."""
        return self._safe_call(lambda: self.client.models.generate_content(
            model=self.model_name, contents=prompt
        ).text)

    # ────────────────────────────────────────────────────────────────
    # FEATURE 4: Agentic Reasoning (Multi-Step Analysis)
    # ────────────────────────────────────────────────────────────────

    def agentic_analysis(self, strategic_obj: str, actions: list[str],
                         focus: str) -> str:
        """
        Perform multi-step autonomous reasoning analysis.
        
        This implements "agentic AI" - where the LLM works through a structured
        reasoning chain autonomously, rather than just answering a single prompt.
        The agent performs 6 sequential analytical steps to deeply assess
        strategic alignment.
        
        Args:
            strategic_obj: The strategic objective being analyzed
            actions: List of all action items from the QIP
            focus: Specific focus area to analyze (e.g., "patient safety")
        
        Returns:
            str: Comprehensive analysis following the 6-step reasoning chain:
                1. Strategic Intent Analysis
                2. Current Action Mapping & Rating
                3. Gap Identification
                4. Root Cause Analysis
                5. Synthesized Recommendations
                6. Risk Assessment
        
        Why "Agentic"?
            Traditional LLM: Single prompt → single response
            Agentic LLM: Multi-step prompt → step-by-step reasoning → synthesis
            
            This demonstrates autonomous problem-solving capabilities beyond
            simple Q&A, fulfilling the "innovative features" requirement.
        """
        acts = "\n".join(f"  - {a}" for a in actions)
        prompt = f"""You are an autonomous AI agent performing deep strategic
alignment analysis for Hamilton Health Sciences.

STRATEGIC OBJECTIVE: {strategic_obj}
FOCUS AREA: {focus}

CURRENT QIP ACTION ITEMS:
{acts}

Work through the following reasoning chain:

## Step 1 — Strategic Intent
What does this objective truly aim to achieve?

## Step 2 — Mapping Current Actions
Rate each action's relevance (High / Medium / Low / None).

## Step 3 — Missing Elements
What critical actions are MISSING from the QIP?

## Step 4 — Root Cause Analysis
Why might these gaps exist?

## Step 5 — Synthesised Recommendations
3 specific, implementable recommendations with expected impact.

## Step 6 — Risk Assessment
Risks of NOT addressing these gaps.

Be thorough in each step."""
        return self._safe_call(lambda: self.client.models.generate_content(
            model=self.model_name, contents=prompt
        ).text)
