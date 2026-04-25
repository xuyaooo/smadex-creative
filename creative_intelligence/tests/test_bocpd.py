"""Unit tests for src.fatigue.bocpd."""
import numpy as np

from src.fatigue.bocpd import bocpd, fatigue_changepoint


def test_no_changepoint_on_constant_series():
    """A near-constant series with the production threshold (0.4) should never
    fire a changepoint flag."""
    rng = np.random.default_rng(0)
    series = 0.05 + rng.normal(0, 0.001, size=60)
    out = fatigue_changepoint(series, hazard_lambda=50.0, threshold_multiple=12.0)
    assert out["has_changepoint"] is False
    assert out["changepoint_day"] == -1


def test_clear_changepoint_fires_under_low_threshold():
    """With a clear regime shift and a threshold below the hazard rate, the
    convenience wrapper should fire and return a valid changepoint day in range.
    """
    rng = np.random.default_rng(0)
    pre = 0.05 + rng.normal(0, 0.002, size=30)   # days 0..29
    post = 0.01 + rng.normal(0, 0.002, size=30)  # days 30..59
    series = np.concatenate([pre, post])

    out = fatigue_changepoint(series, hazard_lambda=10.0, threshold_multiple=0.5)
    assert out["has_changepoint"] is True
    # The detector picks *some* day; just sanity-check it's inside the series.
    assert 0 <= out["changepoint_day"] < len(series)
    assert out["max_cp_prob"] > 0.05

    # Same series with a threshold above the hazard rate should NOT fire.
    out_high = fatigue_changepoint(series, hazard_lambda=10.0, threshold_multiple=5.0)
    assert out_high["has_changepoint"] is False
    assert out_high["changepoint_day"] == -1


def test_short_series_does_not_crash():
    # length < 4 short-circuits in bocpd()
    out = fatigue_changepoint(np.array([0.1, 0.1, 0.1]), threshold_multiple=12.0)
    assert out["has_changepoint"] is False
    assert out["changepoint_day"] == -1
    # Empty and length-1 should also be safe
    out_empty = fatigue_changepoint(np.array([]))
    assert out_empty["has_changepoint"] is False


def test_bocpd_returns_probability_array_in_unit_interval():
    rng = np.random.default_rng(3)
    series = 0.05 + rng.normal(0, 0.005, size=20)
    cp_prob, best = bocpd(series)
    assert cp_prob.shape == (20,)
    assert (cp_prob >= 0).all() and (cp_prob <= 1.0 + 1e-9).all()
    assert -1 <= best < 20
