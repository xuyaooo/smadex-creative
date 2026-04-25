"""Unit tests for src.inference.explainer."""
from src.inference.explainer import (
    FEATURE_LABELS,
    _label,
    counterfactual_suggestion,
    explain_creative,
)


REQUIRED_EXPLAIN_KEYS = {
    "headline", "why_it_works", "what_to_watch",
    "rubric_callouts", "action_line",
    "shap_top_pos", "shap_top_neg",
}


def test_explain_creative_returns_required_keys():
    shap = {
        "early_ctr": 0.12,
        "color_vibrancy": 0.05,
        "early_ipm": -0.04,
        "text_density_visual": -0.02,
    }
    out = explain_creative(
        perf_pred=0.62,
        perf_percentile_vertical=0.71,
        vertical="gaming",
        shap_dict=shap,
        rubric={"hook_clarity": 8, "cta_prominence": 3},
        health={"action": "Scale", "health_score": 78.0, "severity": "healthy"},
    )
    assert REQUIRED_EXPLAIN_KEYS.issubset(out.keys())
    assert "gaming" in out["headline"]
    assert "0.62" in out["headline"]
    # Rubric callouts should mention strong (8) and weak (3) axes
    assert any("Strong" in line for line in out["rubric_callouts"])
    assert any("Weak" in line for line in out["rubric_callouts"])
    # Action sentence should be wired to the supplied health dict
    assert "Scale" in out["action_line"]


def test_label_resolves_known_features_and_prefixes():
    # Direct dictionary hits
    assert _label("early_ctr") == FEATURE_LABELS["early_ctr"]
    assert _label("cta_prominence") == "CTA prominence"
    # Prefix-driven mappings
    assert "gaming" in _label("vertical_gaming")
    assert "rewarded video" in _label("format_rewarded_video")
    assert _label("clip_pc7").startswith("visual style")
    # Unknown -> spaces in place of underscores
    assert _label("some_random_feature") == "some random feature"


def test_counterfactual_skips_strong_axes():
    rubric = {
        "hook_clarity": 9,        # already strong: must be skipped
        "cta_prominence": 3,
        "color_vibrancy": 4,
        "text_density_visual": 2,
    }
    importances = {"cta_prominence": 0.2, "color_vibrancy": 0.1, "text_density_visual": 0.05}
    out = counterfactual_suggestion(rubric, perf_pred=0.5, rubric_importances=importances, n_top=5)
    axes = [r["axis"] for r in out]
    assert "hook_clarity" not in axes
    # Each suggestion should propose a higher target than current
    for r in out:
        assert r["target"] > r["current"]
        assert r["target"] <= 10


def test_counterfactual_with_empty_rubric_returns_empty():
    out = counterfactual_suggestion({}, perf_pred=0.5, rubric_importances={}, n_top=3)
    assert out == []
