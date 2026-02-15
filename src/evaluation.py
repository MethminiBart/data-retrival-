"""
Evaluation Module — compares the system's embedding-based alignment
detection against a manually-defined ground truth to produce
precision, recall, F1 and a confusion matrix.

This module implements the "Testing and Evaluation" requirement from
the coursework, demonstrating that the system produces accurate results.

Evaluation Methodology:
1. Ground Truth: Manual mapping (ALIGNMENT_MAP from knowledge_graph.py)
2. Predicted: Cosine similarity >= threshold
3. Metrics: Precision, Recall, F1-score (industry-standard IR metrics)
4. Confusion Matrix: True/False Positives/Negatives

Key Formulas:
    Precision = TP / (TP + FP)  # Of predicted alignments, how many are correct?
    Recall = TP / (TP + FN)     # Of actual alignments, how many did we find?
    F1 = 2 * (P * R) / (P + R)  # Harmonic mean of Precision and Recall
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.document_processor import get_strategic_texts, get_action_texts
from src.knowledge_graph import ALIGNMENT_MAP

# Ground truth is the manually-defined alignment map
GROUND_TRUTH = ALIGNMENT_MAP


def evaluate(sim_matrix: np.ndarray, threshold: float = 0.45) -> dict:
    """
    Evaluate system performance against ground truth.
    
    This function converts the continuous similarity matrix into binary
    predictions (aligned/not aligned) using a threshold, then compares
    against the manual ground truth mapping.
    
    Args:
        sim_matrix: NxM matrix of cosine similarities (N strategies, M actions)
        threshold: Similarity cutoff for considering alignment (default 0.45)
    
    Returns:
        dict: Comprehensive evaluation results containing:
            - overall: Precision, Recall, F1 across all predictions
            - confusion_matrix: 2x2 matrix [[TN, FP], [FN, TP]]
            - per_strategy: Metrics broken down by strategic objective
            - gt_positives: Total alignments in ground truth
            - pred_positives: Total alignments predicted by system
            - true_pos, false_pos, false_neg, true_neg: Raw counts
    
    Threshold Selection:
        - Too low (e.g., 0.2): High recall, low precision (many false positives)
        - Too high (e.g., 0.7): High precision, low recall (miss real alignments)
        - 0.45 is empirically tuned for balanced F1-score
    
    Why Binary Classification?
        IR systems must make binary decisions (retrieve or don't retrieve).
        Even though cosine similarity is continuous, we need a cutoff to
        determine "aligned" vs "not aligned" for practical use.
    """
    # Get IDs in consistent order
    s_ids = [t["id"] for t in get_strategic_texts()]
    a_ids = [t["id"] for t in get_action_texts()]

    # ────────────────────────────────────────────────────────────────
    # Build ground truth binary matrix from manual mapping
    # ────────────────────────────────────────────────────────────────
    
    gt = np.zeros((len(s_ids), len(a_ids)), dtype=int)  # Start with all 0s
    for i, sid in enumerate(s_ids):
        for aid in GROUND_TRUTH.get(sid, []):  # Get aligned actions for this strategy
            if aid in a_ids:
                gt[i][a_ids.index(aid)] = 1  # Mark as aligned (1)
    
    # ────────────────────────────────────────────────────────────────
    # Convert similarity matrix to binary predictions
    # ────────────────────────────────────────────────────────────────
    
    pred = (sim_matrix >= threshold).astype(int)  # 1 if sim >= threshold, else 0

    # Flatten matrices for sklearn metrics (which expect 1D arrays)
    gt_f = gt.flatten()      # Shape: (N*M,)
    pred_f = pred.flatten()  # Shape: (N*M,)

    # ────────────────────────────────────────────────────────────────
    # Compute overall metrics
    # ────────────────────────────────────────────────────────────────
    
    p = precision_score(gt_f, pred_f, zero_division=0)
    r = recall_score(gt_f, pred_f, zero_division=0)
    f = f1_score(gt_f, pred_f, zero_division=0)
    cm = confusion_matrix(gt_f, pred_f)

    # ────────────────────────────────────────────────────────────────
    # Compute per-strategy metrics (for detailed analysis)
    # ────────────────────────────────────────────────────────────────
    
    per_strategy: dict[str, dict] = {}
    for i, sid in enumerate(s_ids):
        # Get this strategy's row from both matrices
        per_strategy[sid] = {
            "precision": float(precision_score(gt[i], pred[i], zero_division=0)),
            "recall": float(recall_score(gt[i], pred[i], zero_division=0)),
            "f1": float(f1_score(gt[i], pred[i], zero_division=0)),
        }

    # ────────────────────────────────────────────────────────────────
    # Extract confusion matrix components
    # ────────────────────────────────────────────────────────────────
    
    # Confusion matrix layout:
    #           Predicted
    #           0      1
    # Actual 0  TN     FP
    #        1  FN     TP
    
    return {
        "overall": {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "threshold": threshold
        },
        "confusion_matrix": cm.tolist(),  # Convert numpy to list for JSON serialization
        "per_strategy": per_strategy,
        "gt_positives": int(gt_f.sum()),     # Total alignments in ground truth
        "pred_positives": int(pred_f.sum()), # Total alignments predicted
        "true_pos": int(cm[1][1]) if cm.shape[0] > 1 else 0,
        "false_pos": int(cm[0][1]) if cm.shape[0] > 1 else 0,
        "false_neg": int(cm[1][0]) if cm.shape[0] > 1 else 0,
        "true_neg": int(cm[0][0]),
    }


def evaluate_thresholds(sim_matrix: np.ndarray) -> list[dict]:
    """
    Run evaluation at multiple thresholds to find the optimal cutoff.
    
    This function performs a threshold sweep from 0.20 to 0.75 in steps
    of 0.05, computing metrics at each point. Used to create the
    threshold optimization chart on the Evaluation page.
    
    Args:
        sim_matrix: NxM matrix of cosine similarities
    
    Returns:
        list[dict]: List of evaluation results, one per threshold, each containing:
            - threshold: The cutoff value tested
            - precision: Precision at this threshold
            - recall: Recall at this threshold
            - f1: F1-score at this threshold
    
    Use Case:
        Visualize precision/recall tradeoff to justify threshold selection.
        The optimal threshold maximizes F1-score (balance of P and R).
    
    Typical Patterns:
        - Low threshold (0.2): High R, Low P (over-predicts alignment)
        - High threshold (0.7): Low R, High P (under-predicts alignment)
        - Sweet spot (~0.45): Balanced F1
    """
    return [
        {
            "threshold": round(float(t), 2),
            # Extract just the metrics we want to plot (exclude nested data)
            **{k: v for k, v in evaluate(sim_matrix, t)["overall"].items()
               if k != "threshold"},
        }
        for t in np.arange(0.20, 0.75, 0.05)  # Test thresholds: 0.20, 0.25, ..., 0.70
    ]
