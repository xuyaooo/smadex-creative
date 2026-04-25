"""
Creative Health Score 0–100.

A single number a marketer can act on — combines:
  - Fatigue probability (LightGBM classifier on early-window features)
  - Performance percentile within vertical
  - Trajectory: BOCPD changepoint signal on daily CTR
  - Predicted perf_score from XGBoost

Mapping is monotonic, deterministic, and < 1 ms to compute. No LLM at runtime.

Score >= 75 → "Healthy" (scale)
Score 50–75 → "Watch"  (continue)
Score 25–50 → "Risk"   (pivot)
Score < 25  → "Pause"  (kill)

Optional `status_probs` (calibrated classifier output for the 4-class status
head) can override the action selection: a confident top_performer prediction
should drive a Scale recommendation even when the conservative perf_pred holds
the raw health score in the Continue band.
"""
from typing import Dict, Optional


def health_score(
    perf_pred: float,
    perf_percentile_vertical: float,
    fatigue_prob: float,
    has_changepoint: bool,
    days_active: int,
    days_remaining_estimate: int = -1,
    status_probs: Optional[Dict[str, float]] = None,
) -> Dict:
    """Combine signals into a 0-100 score with a recommended action.

    All inputs are numbers we already produce in the pipeline.

    If `status_probs` is provided (keys: "top_performer", "stable", "fatigued",
    "underperformer"), the calibrated class confidences override the
    threshold-only action selection at the margins.
    """
    # Component scores, each in [0, 1]
    perf_component = max(0.0, min(1.0, perf_pred))                   # already roughly [0,1]
    rank_component = max(0.0, min(1.0, perf_percentile_vertical))    # 0=worst, 1=best
    fatigue_component = 1.0 - max(0.0, min(1.0, fatigue_prob))       # invert: low risk=good

    # Trajectory: changepoint detected → moderate negative signal
    # (every running creative has *some* day-to-day variance; we only penalize
    # confirmed regime shifts, and not so hard that a true top performer drops to "Pivot")
    trajectory_component = 1.0 if not has_changepoint else 0.65

    # Time-pressure: days_remaining estimate (if model says "fatigues in 2 days", penalize)
    pressure_component = 1.0
    if days_remaining_estimate >= 0:
        pressure_component = max(0.0, min(1.0, days_remaining_estimate / 14.0))

    # Weighted blend (sum of weights = 1)
    raw = (
        0.30 * perf_component
        + 0.25 * rank_component
        + 0.25 * fatigue_component
        + 0.10 * trajectory_component
        + 0.10 * pressure_component
    )
    score = round(100 * raw, 1)

    # Default: threshold-on-score action selection.
    if score >= 75:
        action, severity = "Scale", "healthy"
    elif score >= 50:
        action, severity = "Continue", "watch"
    elif score >= 25:
        action, severity = "Pivot", "risk"
    else:
        action, severity = "Pause", "critical"

    # Confidence-aware overrides. The calibrated 4-way status classifier sees
    # signals (clip embedding, full feature stack) that the conservative
    # perf_pred regressor under-weights, so a high p(top) should escalate
    # Continue → Scale, and a high p(fatigued) / p(under) should be acted on
    # even when the blended score sits in a benign band.
    if status_probs is not None:
        p_top = float(status_probs.get("top_performer", 0.0))
        p_stable = float(status_probs.get("stable", 0.0))
        p_fatigued = float(status_probs.get("fatigued", 0.0))
        p_under = float(status_probs.get("underperformer", 0.0))

        # Pause beats everything: a confident fatigue call must not be masked
        # by lukewarm percentile / perf_pred.
        if p_fatigued >= 0.5:
            action, severity = "Pause", "critical"
        elif p_under >= 0.5:
            action, severity = "Pivot", "risk"
        elif p_top >= 0.4 and score >= 50:
            action, severity = "Scale", "healthy"
        elif p_top >= 0.55:
            # Very confident top performer — scale even if score < 50,
            # since perf_pred / percentile are known to be conservative.
            action, severity = "Scale", "healthy"
        elif p_stable >= 0.5 and action in ("Pivot",):
            # Stable creatives mis-bucketed below 50 by a soft perf_pred
            # should keep running, not be pivoted.
            action, severity = "Continue", "watch"

    return {
        "health_score": score,
        "action": action,
        "severity": severity,
        "components": {
            "performance": round(100 * perf_component, 1),
            "rank_in_vertical": round(100 * rank_component, 1),
            "fatigue_resistance": round(100 * fatigue_component, 1),
            "trajectory": round(100 * trajectory_component, 1),
            "time_pressure": round(100 * pressure_component, 1),
        },
        "days_active": days_active,
        "days_remaining_estimate": days_remaining_estimate,
    }
