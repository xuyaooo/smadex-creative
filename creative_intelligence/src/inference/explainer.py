"""
Templated, marketer-readable explanations from SHAP + rubric features.

No LLM at runtime: all phrasing is deterministic, filled from numbers we
already produce. Pattern after AutoCO (Alibaba WWW 2021) and Menon-Vondrick
verbalized concepts (ICLR 2023).
"""
from typing import Dict, List, Optional, Tuple


# Friendly names for raw feature columns
FEATURE_LABELS: Dict[str, str] = {
    "early_ctr": "early-life CTR",
    "early_ipm": "early installs-per-mille",
    "early_cvr": "early conversion rate",
    "early_roas": "early return on ad spend",
    "early_ctr_slope": "CTR trajectory in first week",
    "early_impressions_mean": "early impressions volume",
    "early_video_completion_rate": "video completion rate",
    "early_viewable_sum": "viewable impressions",
    "campaign_duration": "campaign length",
    "daily_budget_usd": "daily budget",
    "aspect_ratio": "aspect ratio",
    "duration_sec": "creative duration",
    "novelty_visual": "visual novelty",
    "cta_prominence": "CTA prominence",
    "cta_contrast": "CTA contrast",
    "color_vibrancy": "color vibrancy",
    "color_warmth": "color warmth",
    "text_density_visual": "text density",
    "product_focus": "product focus",
    "scene_realism": "photo realism",
    "emotion_intensity": "emotional intensity",
    "composition_balance": "composition balance",
    "brand_visibility": "brand visibility",
    "urgency_signal": "urgency signal",
    "playfulness": "playfulness",
    "hook_clarity": "hook clarity",
}


def _label(feat: str) -> str:
    """Resolve a raw feature name to a marketer-readable phrase."""
    if feat in FEATURE_LABELS:
        return FEATURE_LABELS[feat]
    if feat.startswith("vertical_"):
        return f"being a {feat.removeprefix('vertical_')} ad"
    if feat.startswith("format_"):
        return f"the {feat.removeprefix('format_').replace('_', ' ')} format"
    if feat.startswith("dominant_color_"):
        return f"a {feat.removeprefix('dominant_color_')} dominant color"
    if feat.startswith("objective_"):
        return f"the {feat.removeprefix('objective_').replace('_', ' ')} objective"
    if feat.startswith("emotional_tone_"):
        return f"a {feat.removeprefix('emotional_tone_')} tone"
    if feat.startswith("kpi_goal_"):
        return f"the {feat.removeprefix('kpi_goal_')} KPI goal"
    if feat.startswith("clip_pc"):
        return "visual style (image embedding)"
    return feat.replace("_", " ")


def _trim_shap(shap_dict: Dict[str, float], top_k: int = 5) -> List[Tuple[str, float]]:
    return sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]


def explain_creative(
    perf_pred: float,
    perf_percentile_vertical: float,
    vertical: str,
    shap_dict: Dict[str, float],
    rubric: Optional[Dict[str, int]] = None,
    health: Optional[Dict] = None,
) -> Dict:
    """Build a structured explanation dict the demo can render."""
    # Top SHAP contributors split by sign
    top_pos = [(f, v) for f, v in _trim_shap(shap_dict, 8) if v > 0][:3]
    top_neg = [(f, v) for f, v in _trim_shap(shap_dict, 8) if v < 0][:3]

    pct = round(100 * perf_percentile_vertical)
    headline = (
        f"Predicted perf score: {perf_pred:.2f} "
        f"({pct}th percentile in {vertical})."
    )

    why_works = [f"{_label(f)} (+{v:+.3f})" for f, v in top_pos] or ["no strong positive drivers"]
    why_risks = [f"{_label(f)} ({v:+.3f})" for f, v in top_neg] or ["no clear weaknesses"]

    rubric_lines: List[str] = []
    if rubric:
        ranked = sorted(rubric.items(), key=lambda kv: kv[1], reverse=True)
        top = ranked[:2]
        bot = ranked[-2:]
        for k, v in top:
            if v >= 7:
                rubric_lines.append(f"Strong: {_label(k)} ({v}/10)")
        for k, v in bot:
            if v <= 3:
                rubric_lines.append(f"Weak: {_label(k)} ({v}/10)")

    # Action sentence
    if health is not None:
        action_line = (
            f"Recommended action: **{health['action']}** "
            f"(Health {health['health_score']}/100, {health['severity']})."
        )
    else:
        action_line = ""

    return {
        "headline": headline,
        "why_it_works": why_works,
        "what_to_watch": why_risks,
        "rubric_callouts": rubric_lines,
        "action_line": action_line,
        "shap_top_pos": top_pos,
        "shap_top_neg": top_neg,
    }


def counterfactual_suggestion(
    rubric: Dict[str, int],
    perf_pred: float,
    rubric_importances: Dict[str, float],
    n_top: int = 3,
) -> List[Dict]:
    """Cheap counterfactual: rank rubric axes by importance × room-to-grow.

    For each weak rubric axis (score <= 4) that is also in the top-N important
    features, suggest raising it. Returns a list of {axis, current, target, why}.
    No model retraining at query time — uses precomputed importance weights.
    """
    rows = []
    # Default floor: every rubric axis carries some weight even if not in top splits.
    floor = 0.005
    for axis, score in rubric.items():
        if score >= 7:  # already strong, skip
            continue
        importance = max(rubric_importances.get(axis, 0.0), floor)
        # Heuristic: how much "room to grow" weighted by importance
        priority = (8 - score) * importance
        target = min(10, score + 4)
        rows.append({
            "axis": axis,
            "axis_label": _label(axis),
            "current": score,
            "target": target,
            "importance": round(importance, 4),
            "priority": round(priority, 4),
            "advice": (
                f"Raise {_label(axis)} from {score}/10 to ~{target}/10."
            ),
        })

    rows.sort(key=lambda r: -r["priority"])
    return rows[:n_top]
