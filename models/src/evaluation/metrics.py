"""Evaluation metrics for all model components."""

import json
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, r2_score, roc_auc_score,
)

STATUS_LABELS = ["top_performer", "stable", "fatigued", "underperformer"]


def evaluate_tabular(y_true_perf, y_pred_perf, y_true_status, y_pred_status) -> Dict:
    return {
        "perf_mae": float(mean_absolute_error(y_true_perf, y_pred_perf)),
        "perf_r2": float(r2_score(y_true_perf, y_pred_perf)),
        "status_report": classification_report(
            y_true_status, y_pred_status, target_names=STATUS_LABELS, output_dict=True
        ),
    }


def evaluate_fatigue(y_true_binary, y_pred_proba, y_true_day, y_pred_day, fatigued_mask) -> Dict:
    results = {
        "auc_roc": float(roc_auc_score(y_true_binary, y_pred_proba)),
        "classification_report": classification_report(
            y_true_binary, (y_pred_proba > 0.5).astype(int),
            target_names=["not_fatigued", "fatigued"],
            output_dict=True,
        ),
    }
    if fatigued_mask.sum() > 0:
        results["fatigue_day_mae"] = float(
            mean_absolute_error(y_true_day[fatigued_mask], y_pred_day[fatigued_mask])
        )
        results["fatigue_day_within2d_acc"] = float(
            np.mean(np.abs(y_true_day[fatigued_mask] - y_pred_day[fatigued_mask]) <= 2)
        )
    return results


def evaluate_vlm_labels(predictions: List[Dict], references: List[Dict]) -> Dict:
    """Evaluate student VLM outputs against teacher references."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    except ImportError:
        scorer = None

    parse_success = 0
    rouge_scores = []
    required_keys = ["performance_summary", "visual_strengths", "visual_weaknesses",
                     "fatigue_risk_reason", "top_recommendation"]

    for pred, ref in zip(predictions, references):
        if isinstance(pred, dict) and all(k in pred for k in required_keys):
            parse_success += 1
            if scorer is not None:
                score = scorer.score(
                    ref.get("performance_summary", ""),
                    pred.get("performance_summary", ""),
                )
                rouge_scores.append(score["rougeL"].fmeasure)

    n = len(predictions)
    result = {"json_parse_rate": parse_success / n if n > 0 else 0.0, "n": n}
    if rouge_scores:
        result["rouge_l_mean"] = float(np.mean(rouge_scores))
    return result


def evaluate_retrieval(retrieved_ids: List[int], master_df) -> Dict:
    statuses = master_df.set_index("creative_id")["creative_status"]
    good = {"top_performer", "stable"}
    precision_at_k = sum(1 for cid in retrieved_ids if statuses.get(cid) in good) / len(retrieved_ids)
    return {"precision_at_k": precision_at_k, "k": len(retrieved_ids)}
