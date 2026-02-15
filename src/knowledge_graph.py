"""
Knowledge Graph — builds a NetworkX directed graph showing how strategic
objectives, foundational commitments, values and action items relate,
and renders it as an interactive PyVis HTML visualisation.

This module implements two key coursework requirements:
1. Knowledge graph visualization (innovative feature)
2. Ontology-based mapping (ground truth for evaluation)

The graph structure represents the strategic landscape:
- Strategic Aims (LEAD, BUILD, SERVE, LEARN) - top-level objectives
- Foundational Commitments - underpinning principles
- Values (Collaboration, Compassion, etc.) - cultural drivers
- Action Items - concrete QIP initiatives

Edges represent relationships:
- Values → Strategic Aims: "guides"
- Commitments → Strategic Aims: "underpins"
- Strategic Aims → Actions: "supports" (defined by ALIGNMENT_MAP)
"""

import tempfile
import networkx as nx
from pyvis.network import Network

from src.document_processor import (
    STRATEGIC_OBJECTIVES, ACTION_ITEMS, FOUNDATIONAL_COMMITMENTS, VALUES,
)

# ────────────────────────────────────────────────────────────────────
# Ground Truth Alignment Map (Ontology-Based Mapping)
# ────────────────────────────────────────────────────────────────────

ALIGNMENT_MAP: dict[str, list[str]] = {
    """
    Manual mapping of strategic objectives to action items.
    
    This serves as the **ground truth** for system evaluation.
    Created by manually analyzing both documents and determining
    which action items directly support each strategic objective.
    
    Structure:
        {strategic_code: [action_id1, action_id2, ...]}
    
    Example:
        "LEAD": ["QIP-1", "QIP-2"] means QIP-1 and QIP-2 support
        the "Lead in Research & Innovation" strategic aim.
    
    This mapping is used for:
        1. Evaluation metrics (precision, recall, F1)
        2. Knowledge graph edge creation
        3. Validating the embedding-based similarity scores
    """
    "LEAD":   ["QIP-1", "QIP-2", "S-1", "S-2", "PH-2", "AF-3"],
    "BUILD":  ["PH-1", "PH-2", "PC-2", "AF-3"],
    "SERVE":  ["QIP-1", "QIP-2", "QIP-3", "QIP-4", "QIP-5", "QIP-6",
               "QIP-7", "AF-1", "AF-2", "PE-1", "PE-2", "PE-3", "PC-1", "EI-2"],
    "LEARN":  ["PX-1", "EI-1", "S-1", "S-2"],
    "FC-HEA": ["QIP-5", "QIP-7", "EI-1", "EI-2", "PE-3"],  # Foundational Commitment: Health Equity
    "FC-IND": ["EI-3", "QIP-7"],  # Foundational Commitment: Indigenous Health
    "FC-RES": ["QIP-1", "QIP-2", "S-2", "PH-2", "AF-1"],  # Foundational Commitment: Research Excellence
}


def build_graph() -> nx.DiGraph:
    """
    Construct a directed graph representing the strategic landscape.
    
    Returns:
        nx.DiGraph: NetworkX directed graph with 4 node types:
            - strategic_aim: Blue diamonds (size 35)
            - commitment: Purple dots (size 28)
            - value: Teal dots (size 20)
            - action: Orange dots (size 16)
    
    Node Attributes:
        - label: Display name
        - node_type: Category (strategic_aim, commitment, value, action)
        - description: Full text description
        - color: Hex color code for visualization
        - size: Node size for PyVis rendering
    
    Edge Types:
        - guides: Values → Strategic Aims (light blue)
        - underpins: Commitments → Strategic Aims (light purple)
        - supports: Strategic Aims → Actions (blue, from ALIGNMENT_MAP)
    
    Design Decision:
        We use a directed graph (DiGraph) rather than an undirected graph
        because relationships have direction:
        - Values guide strategies (not vice versa)
        - Strategies are supported by actions (not vice versa)
    """
    G = nx.DiGraph()

    # Add strategic aim nodes (blue diamonds, largest size)
    for o in STRATEGIC_OBJECTIVES:
        G.add_node(
            o.code,  # Node ID (e.g., "LEAD", "BUILD")
            label=o.name,  # Display name
            node_type="strategic_aim",
            description=o.description,
            color="#1E88E5",  # Blue
            size=35  # Largest nodes
        )

    # Add foundational commitment nodes (purple, medium size)
    for fc in FOUNDATIONAL_COMMITMENTS:
        G.add_node(
            fc["code"],  # e.g., "FC-HEA"
            label=fc["name"],
            node_type="commitment",
            description=fc["description"],
            color="#7B1FA2",  # Purple
            size=28
        )

    # Add value nodes (teal, smaller size)
    for name, desc in VALUES.items():
        # Create ID from first 4 letters (e.g., "VAL-COLL" for Collaboration)
        vid = f"VAL-{name[:4].upper()}"
        G.add_node(
            vid,
            label=name,
            node_type="value",
            description=desc,
            color="#00897B",  # Teal
            size=20
        )

    # Add action item nodes (orange, smallest size)
    lookup = {a.id: a for a in ACTION_ITEMS}  # Quick lookup dict
    for a in ACTION_ITEMS:
        G.add_node(
            a.id,  # e.g., "QIP-1", "AF-2"
            label=a.title[:35],  # Truncate long titles for display
            node_type="action",
            section=a.section,  # QIP Priority, Access & Flow, etc.
            description=a.description,
            color="#FF6D00",  # Orange
            size=16  # Smallest nodes
        )

    # ────────────────────────────────────────────────────────────────
    # Add edges: Alignment relationships (strategic aims → actions)
    # ────────────────────────────────────────────────────────────────
    
    for src, targets in ALIGNMENT_MAP.items():
        for tgt in targets:
            if tgt in lookup:  # Verify action exists before adding edge
                G.add_edge(
                    src, tgt,
                    relationship="supports",  # Semantic label
                    color="#90CAF9",  # Light blue
                    weight=2  # Thicker line (emphasizes main alignment)
                )

    # ────────────────────────────────────────────────────────────────
    # Add edges: Values → Strategic Aims (guides)
    # ────────────────────────────────────────────────────────────────
    
    for o in STRATEGIC_OBJECTIVES:
        for name in VALUES:
            G.add_edge(
                f"VAL-{name[:4].upper()}", o.code,
                relationship="guides",
                color="#B2DFDB",  # Very light teal
                weight=1  # Thinner line
            )

    # ────────────────────────────────────────────────────────────────
    # Add edges: Commitments → Strategic Aims (underpins)
    # ────────────────────────────────────────────────────────────────
    
    for fc in FOUNDATIONAL_COMMITMENTS:
        for o in STRATEGIC_OBJECTIVES:
            G.add_edge(
                fc["code"], o.code,
                relationship="underpins",
                color="#CE93D8",  # Light purple
                weight=1
            )

    return G


def render_pyvis(G: nx.DiGraph) -> str:
    """
    Convert NetworkX graph to interactive HTML visualization using PyVis.
    
    PyVis creates a force-directed layout with:
    - Physics simulation (nodes repel/attract based on edges)
    - Interactive navigation (drag, zoom, click for details)
    - Hover tooltips showing node descriptions
    
    Args:
        G: NetworkX directed graph from build_graph()
    
    Returns:
        str: Path to temporary HTML file containing the visualization
    
    Physics Parameters (Barnes-Hut):
        - gravity: -3000 (negative = repulsion, keeps nodes separated)
        - central_gravity: 0.3 (slight pull toward center)
        - spring_length: 200 (ideal edge length in pixels)
    
    Node Rendering:
        - Diamonds for strategic aims (more prominent)
        - Dots for everything else
        - Tooltips show full label + description on hover
    """
    # Initialize PyVis network
    net = Network(
        height="620px",
        width="100%",
        directed=True,  # Show arrow directions
        bgcolor="#FAFAFA",  # Light gray background
        font_color="#212121"  # Almost black text
    )
    
    # Configure physics (Barnes-Hut algorithm for large graphs)
    net.barnes_hut(
        gravity=-3000,  # Strong repulsion keeps nodes apart
        central_gravity=0.3,  # Weak pull toward center
        spring_length=200  # Target distance between connected nodes
    )

    # Add all nodes with visual properties
    for nid, d in G.nodes(data=True):
        net.add_node(
            nid,
            label=d.get("label", nid),
            color=d.get("color", "#999"),  # Default gray if missing
            size=d.get("size", 15),
            title=f"{d.get('label', nid)}\n{d.get('description', '')}",  # Hover text
            shape="diamond" if d.get("node_type") == "strategic_aim" else "dot",
        )

    # Add all edges with visual properties
    for u, v, d in G.edges(data=True):
        net.add_edge(
            u, v,
            title=d.get("relationship", ""),  # Hover text on edge
            color=d.get("color", "#CCC"),  # Default light gray
            width=d.get("weight", 1)  # Line thickness
        )

    # Save to temporary file (Streamlit will display it)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w")
    net.save_graph(tmp.name)
    return tmp.name


def graph_stats(G: nx.DiGraph) -> dict:
    """
    Compute summary statistics for the knowledge graph.
    
    Args:
        G: NetworkX directed graph
    
    Returns:
        dict: Statistics including:
            - nodes: Total node count
            - edges: Total edge count
            - strategic_aims: Count of strategic aim nodes
            - actions: Count of action item nodes
            - commitments: Count of foundational commitment nodes
            - values: Count of value nodes
            - density: Graph density (0=sparse, 1=complete graph)
    
    Density Formula:
        density = |E| / (|V| * (|V| - 1))
        where |E| = edges, |V| = nodes
        
        Interpretation:
        - 0.0: No connections
        - 1.0: Every node connected to every other node
        - Typical real-world networks: 0.01 - 0.1
    """
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "strategic_aims": sum(1 for _, d in G.nodes(data=True) 
                             if d.get("node_type") == "strategic_aim"),
        "actions": sum(1 for _, d in G.nodes(data=True) 
                      if d.get("node_type") == "action"),
        "commitments": sum(1 for _, d in G.nodes(data=True) 
                          if d.get("node_type") == "commitment"),
        "values": sum(1 for _, d in G.nodes(data=True) 
                     if d.get("node_type") == "value"),
        "density": round(nx.density(G), 4),  # Round to 4 decimal places
    }
