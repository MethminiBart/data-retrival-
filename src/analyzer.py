"""
Synchronization Analyzer — computes embeddings, stores them in ChromaDB,
and calculates alignment scores between strategic objectives and action items.

This is the CORE ENGINE of the ISPS system. It implements:
1. Semantic embedding generation (sentence-transformers)
2. Vector storage and retrieval (ChromaDB)
3. Cosine similarity computation (sklearn)
4. Alignment scoring algorithms (overall, per-strategy, gap analysis)

Data Flow:
    1. Load strategic texts and action texts from document_processor
    2. Encode both using all-MiniLM-L6-v2 → 384-dimensional vectors
    3. Store vectors in ChromaDB for semantic search
    4. Compute similarity matrix (strategies × actions)
    5. Derive metrics: overall score, per-strategy scores, gaps

Performance:
    - Caching: Streamlit caches this class with @st.cache_resource
    - Load time: ~8-12 seconds first run, instant thereafter
    - Model size: ~80MB (all-MiniLM-L6-v2)
    - Vector DB: In-memory (ChromaDB ephemeral client)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

from src.document_processor import (
    get_strategic_texts,
    get_action_texts,
    extract_text_from_pdf,
    chunk_text,
)


class SyncAnalyzer:
    """
    Main analysis engine — embeds documents and computes sync scores.
    
    This class is instantiated once per Streamlit session (cached) and
    provides all the data for the dashboard visualizations.
    
    Attributes:
        model: SentenceTransformer embedding model (local, no API)
        chroma: ChromaDB client (in-memory)
        strategic_texts: List of strategic objective dicts with id, text, type
        action_texts: List of action item dicts with id, text, section
        _s_emb: Strategic text embeddings (shape: [N, 384])
        _a_emb: Action text embeddings (shape: [M, 384])
        _sim: Similarity matrix cache (shape: [N, M])
        s_col: ChromaDB collection for strategic plan
        a_col: ChromaDB collection for action plan
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the analyzer: load model, encode texts, populate vector DB.
        
        Args:
            model_name: HuggingFace model identifier for sentence embeddings
                       Default: all-MiniLM-L6-v2 (384-dim, 80MB, good balance)
        
        Initialization Steps:
            1. Download/load SentenceTransformer model from HuggingFace
            2. Initialize ChromaDB ephemeral client (in-memory)
            3. Load strategic and action texts from document_processor
            4. Encode all texts to vectors (this is the slow step)
            5. Create ChromaDB collections and insert vectors
            6. Prepare similarity matrix (computed lazily on first access)
        
        Model Choice:
            - all-MiniLM-L6-v2: Small, fast, free, good quality
            - Alternatives considered:
                * all-mpnet-base-v2: Higher quality but slower
                * text-embedding-3-small (OpenAI): Requires API, costs money
        
        Why ChromaDB?
            - Simple setup (no external server)
            - Built-in cosine similarity search
            - Perfect for RAG retrieval in chat feature
        """
        # Load embedding model (downloads ~80MB on first run to .hf_cache/)
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB (ephemeral = in-memory, no persistence)
        self.chroma = chromadb.Client()

        # Load structured texts from document processor
        self.strategic_texts = get_strategic_texts()  # [{id, text, type}, ...]
        self.action_texts = get_action_texts()        # [{id, text, section}, ...]

        # ────────────────────────────────────────────────────────────────
        # Encode texts to embeddings
        # ────────────────────────────────────────────────────────────────
        
        # Extract just the text strings for encoding
        s_strings = [t["text"] for t in self.strategic_texts]
        a_strings = [t["text"] for t in self.action_texts]

        # Generate embeddings (shape: [num_texts, 384])
        # show_progress_bar=False to avoid cluttering Streamlit output
        self._s_emb = self.model.encode(s_strings, show_progress_bar=False)
        self._a_emb = self.model.encode(a_strings, show_progress_bar=False)
        
        # Similarity matrix computed lazily (only when first accessed)
        self._sim: np.ndarray | None = None

        # ────────────────────────────────────────────────────────────────
        # ChromaDB setup: Create collections and populate with embeddings
        # ────────────────────────────────────────────────────────────────
        
        # Delete collections if they exist (fresh start each session)
        for name in ("strategic_plan", "action_plan"):
            try:
                self.chroma.delete_collection(name)
            except Exception:
                pass  # Collection doesn't exist yet, that's fine

        # Create new collections
        self.s_col = self.chroma.create_collection("strategic_plan")
        self.a_col = self.chroma.create_collection("action_plan")

        # Populate strategic plan collection
        self.s_col.add(
            documents=s_strings,              # Full text for retrieval
            embeddings=self._s_emb.tolist(),  # Convert numpy to list
            ids=[t["id"] for t in self.strategic_texts],  # e.g., "LEAD", "BUILD"
            metadatas=[{"type": t["type"]} for t in self.strategic_texts],  # For filtering
        )
        
        # Populate action plan collection
        self.a_col.add(
            documents=a_strings,
            embeddings=self._a_emb.tolist(),
            ids=[t["id"] for t in self.action_texts],  # e.g., "QIP-1", "AF-2"
            metadatas=[{"section": t["section"]} for t in self.action_texts],
        )

    # ────────────────────────────────────────────────────────────────
    # Similarity Matrix (Core Data Structure)
    # ────────────────────────────────────────────────────────────────

    @property
    def similarity_matrix(self) -> np.ndarray:
        """
        Lazy-computed cosine similarity matrix between strategies and actions.
        
        Returns:
            np.ndarray: Shape [N, M] where N = strategic items, M = action items
                       Each cell [i, j] = cosine similarity between strategy i and action j
                       Values range from 0 (orthogonal) to 1 (identical meaning)
        
        Caching:
            Computed only once, then cached in self._sim for repeated access.
            This is critical because the matrix is used by:
            - overall_score()
            - strategy_scores()
            - alignment_details()
            - gap_analysis()
            - All dashboard visualizations
        
        Cosine Similarity Formula:
            sim(A, B) = (A · B) / (||A|| * ||B||)
            where A and B are embedding vectors
            
            Interpretation:
            - 1.0: Vectors point in same direction (semantically identical)
            - 0.5: Moderate similarity
            - 0.0: Vectors are orthogonal (unrelated)
            - Negative values theoretically possible but rare with embeddings
        """
        if self._sim is None:
            # Compute cosine similarity between all strategy-action pairs
            # sklearn's cosine_similarity handles normalization automatically
            self._sim = cosine_similarity(self._s_emb, self._a_emb)
        return self._sim

    # ────────────────────────────────────────────────────────────────
    # Scoring Methods (Business Logic)
    # ────────────────────────────────────────────────────────────────

    def overall_score(self) -> float:
        """
        Compute the overall synchronization score (0-100).
        
        Algorithm:
            For each strategic objective:
                1. Find its best-matching action (highest similarity)
                2. Add that score to running total
            3. Average across all strategic objectives
            4. Convert to percentage
        
        Returns:
            float: Overall sync score as percentage (0-100)
                  e.g., 62.4 means 62.4% average alignment
        
        Interpretation:
            - 80-100%: Excellent alignment (most strategies well-supported)
            - 60-80%: Good alignment (room for improvement)
            - 40-60%: Moderate alignment (significant gaps exist)
            - 0-40%: Poor alignment (major disconnect between plans)
        
        Why "Best Match" Instead of Average?
            Design decision: A strategic objective only needs ONE strong
            supporting action to be considered aligned. Using average would
            penalize strategies with focused actions.
        """
        # For each strategy row, take the max similarity across all actions
        # Then average those maxima across all strategies
        # Multiply by 100 to convert to percentage
        return float(self.similarity_matrix.max(axis=1).mean() * 100)

    def strategy_scores(self) -> dict[str, dict]:
        """
        Compute alignment scores broken down by strategic objective.
        
        Returns:
            dict: Mapping of strategic ID to its score dict:
                {
                    "LEAD": {
                        "name": "LEAD",
                        "best_match": 71.2,      # Highest action similarity (%)
                        "average": 45.3,         # Mean across all actions (%)
                        "top3_avg": 65.8         # Mean of top 3 actions (%)
                    },
                    ...
                }
        
        Use Cases:
            - Per-strategy bar chart on home page
            - Radar chart on Sync Analysis page
            - Strategy Deep Dive page selection
        
        Why Three Metrics?
            - best_match: Shows maximum potential alignment
            - average: Shows overall action portfolio relevance
            - top3_avg: Balanced view (not too optimistic or pessimistic)
                       This is what we display in most visualizations
        """
        out: dict[str, dict] = {}
        for i, s in enumerate(self.strategic_texts):
            # Get this strategy's row from similarity matrix
            row = self.similarity_matrix[i]  # Shape: (M,) where M = num actions
            
            out[s["id"]] = {
                "name": s["id"],
                "best_match": float(row.max() * 100),           # Max similarity
                "average": float(row.mean() * 100),             # Mean similarity
                "top3_avg": float(np.sort(row)[-3:].mean() * 100),  # Top 3 mean
            }
        return out

    def alignment_details(self) -> list[dict]:
        """
        Get detailed alignment information for each strategic objective.
        
        Returns:
            list[dict]: One entry per strategic objective containing:
                {
                    "strategic_id": "LEAD",
                    "type": "strategic_aim",
                    "score": 65.8,              # Top-3 average
                    "top_actions": [            # 5 best-aligned actions
                        {"action_id": "QIP-1", "section": "QIP Priority", "similarity": 0.71},
                        {"action_id": "S-2", "section": "Safety", "similarity": 0.68},
                        ...
                    ],
                    "weak_actions": [           # 3 worst-aligned actions
                        {"action_id": "PE-3", "section": "Provider Experience", "similarity": 0.12},
                        ...
                    ]
                }
        
        Use Case:
            Displayed on Strategy Deep Dive page to show which specific
            actions are most/least aligned with each strategy.
        
        Why Top 5 and Bottom 3?
            - Top 5: Enough to see pattern of strong alignments
            - Bottom 3: Just for awareness, not as actionable
        """
        details = []
        for i, s in enumerate(self.strategic_texts):
            row = self.similarity_matrix[i]
            
            # Sort all actions by similarity to this strategy
            scored = sorted(
                [
                    {
                        "action_id": self.action_texts[j]["id"],
                        "section": self.action_texts[j]["section"],
                        "similarity": float(row[j])
                    }
                    for j in range(len(self.action_texts))
                ],
                key=lambda x: x["similarity"],
                reverse=True  # Highest first
            )
            
            details.append({
                "strategic_id": s["id"],
                "type": s["type"],
                "score": float(np.sort(row)[-3:].mean() * 100),  # Top-3 avg score
                "top_actions": scored[:5],   # Best 5
                "weak_actions": scored[-3:],  # Worst 3
            })
        return details

    def gap_analysis(self, threshold: float = 0.45) -> dict:
        """
        Identify strategic objectives with weak action support (gaps).
        
        Args:
            threshold: Minimum similarity for considering adequate alignment
                      Default 0.45 based on evaluation F1 optimization
        
        Returns:
            dict: {
                "gaps": [               # Strategies below threshold
                    {"id": "BUILD", "best_sim": 0.38, "text": "We will build..."},
                    ...
                ],
                "strengths": [          # Strategies above threshold
                    {"id": "SERVE", "best_sim": 0.71, "text": "We will put..."},
                    ...
                ]
            }
        
        Gap Definition:
            A strategic objective has a "gap" if its BEST action alignment
            is below the threshold. This means even the most relevant action
            isn't aligned enough.
        
        Business Value:
            Gaps indicate areas where the action plan doesn't adequately
            support strategic priorities. These should be addressed by:
            - Adding new action items
            - Modifying existing actions to better align
            - Revising strategic objectives if they're unrealistic
        """
        gaps, strengths = [], []
        
        for i, s in enumerate(self.strategic_texts):
            # Get best alignment score for this strategy
            best = float(self.similarity_matrix[i].max())
            
            # Create entry with ID, score, and truncated text
            entry = {
                "id": s["id"],
                "best_sim": best,
                "text": s["text"][:150]  # Truncate for display
            }
            
            # Classify as gap or strength
            if best < threshold:
                gaps.append(entry)
            else:
                strengths.append(entry)
        
        return {"gaps": gaps, "strengths": strengths}

    # ────────────────────────────────────────────────────────────────
    # ChromaDB Query Methods (RAG Support)
    # ────────────────────────────────────────────────────────────────

    def query(self, text: str, collection: str = "action_plan", n: int = 5) -> list[dict]:
        """
        Semantic search: find most similar documents to a query.
        
        This implements the "retrieval" part of RAG (Retrieval-Augmented
        Generation). Given a user question, find the most relevant document
        chunks to provide as context to the LLM.
        
        Args:
            text: Query text (e.g., user question or search term)
            collection: Which collection to search ("action_plan" or "strategic_plan")
            n: Number of results to return (default 5)
        
        Returns:
            list[dict]: Top N most similar documents, each containing:
                {
                    "id": "QIP-2",
                    "text": "Reduce workplace violence...",
                    "distance": 0.234  # Lower = more similar (cosine distance)
                }
        
        How It Works:
            1. Encode query text to embedding vector
            2. ChromaDB computes cosine similarity with all stored vectors
            3. Returns top N matches sorted by similarity
        
        Distance vs. Similarity:
            ChromaDB returns "distance" (lower is better)
            Distance = 1 - cosine_similarity
            So distance 0.2 = similarity 0.8
        """
        # Encode query to vector
        emb = self.model.encode([text], show_progress_bar=False).tolist()
        
        # Select collection
        col = self.a_col if collection == "action_plan" else self.s_col
        
        # Query ChromaDB (returns top N by cosine similarity)
        res = col.query(query_embeddings=emb, n_results=n)
        
        # Format results as list of dicts
        return [
            {
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "distance": res["distances"][0][i]
            }
            for i in range(len(res["ids"][0]))
        ]

    def add_pdf_chunks(self, path: str, source: str) -> int:
        """
        Extract text from PDF, chunk it, and add to vector database.
        
        This method allows adding additional documents beyond the hard-coded
        strategic and action texts. Currently not used in main app flow, but
        available for extensibility.
        
        Args:
            path: File path to PDF
            source: "strategic" or "action" (determines which collection)
        
        Returns:
            int: Number of chunks added
        
        Chunking Strategy:
            - Chunk size: 500 characters
            - Overlap: 100 characters
            - Rationale: Balance between context (longer) and precision (shorter)
        
        Use Case:
            If you wanted to add supplementary documents (e.g., previous year's
            QIP, board meeting minutes), you could use this method to index them.
        """
        # Extract full text from PDF
        text = extract_text_from_pdf(path)
        
        # Split into overlapping chunks
        chunks = chunk_text(text)
        
        # Encode all chunks to embeddings
        embs = self.model.encode(chunks, show_progress_bar=False).tolist()
        
        # Select target collection
        col = self.s_col if source == "strategic" else self.a_col
        
        # Add to ChromaDB with generated IDs
        col.add(
            documents=chunks,
            embeddings=embs,
            ids=[f"{source}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{"source": source, "type": "pdf_chunk"} for _ in chunks],
        )
        
        return len(chunks)
