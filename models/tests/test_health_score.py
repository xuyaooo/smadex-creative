"""Unit tests for src.fatigue.health_score.health_score."""
from src.fatigue.health_score import health_score


EXPECTED_KEYS = {
    "health_score", "action", "severity", "components",
    "days_active", "days_remaining_estimate",
}
EXPECTED_COMPONENT_KEYS = {
    "performance", "rank_in_vertical", "fatigue_resistance",
    "trajectory", "time_pressure",
}
VALID_ACTIONS = {"Scale", "Continue", "Pivot", "Pause"}


def test_high_perf_high_rank_low_fatigue_is_healthy():
    h = health_score(
        perf_pred=0.9,
        perf_percentile_vertical=0.95,
        fatigue_prob=0.05,
        has_changepoint=False,
        days_active=10,
        days_remaining_estimate=14,
    )
    assert h["action"] in {"Scale", "Continue"}
    assert h["health_score"] >= 60


def test_low_perf_high_fatigue_no_runway_is_pause():
    h = health_score(
        perf_pred=0.1,
        perf_percentile_vertical=0.05,
        fatigue_prob=0.95,
        has_changepoint=True,
        days_active=30,
        days_remaining_estimate=0,
    )
    assert h["action"] == "Pause"
    assert h["health_score"] < 30


def test_mid_inputs_yields_continue_or_pivot():
    h = health_score(
        perf_pred=0.5,
        perf_percentile_vertical=0.5,
        fatigue_prob=0.5,
        has_changepoint=False,
        days_active=10,
        days_remaining_estimate=7,
    )
    assert h["action"] in {"Pivot", "Continue"}


def test_health_score_dict_has_all_expected_keys():
    h = health_score(
        perf_pred=0.5,
        perf_percentile_vertical=0.5,
        fatigue_prob=0.5,
        has_changepoint=False,
        days_active=5,
    )
    assert EXPECTED_KEYS.issubset(h.keys())
    assert EXPECTED_COMPONENT_KEYS.issubset(h["components"].keys())
    assert h["action"] in VALID_ACTIONS
    # score must be in [0, 100]
    assert 0.0 <= h["health_score"] <= 100.0
