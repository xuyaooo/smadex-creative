"""Unit tests for src.data.early_features.compute_early_features."""
import numpy as np
import pandas as pd

from src.data.early_features import compute_early_features


DAILY_COLS = [
    "creative_id", "days_since_launch", "country", "os",
    "impressions", "viewable_impressions", "clicks", "conversions",
    "spend_usd", "revenue_usd", "video_completions",
]


def _empty_daily() -> pd.DataFrame:
    return pd.DataFrame({c: [] for c in DAILY_COLS})


def _toy_daily(n_days: int = 7, cid: int = 1, with_imps: bool = True) -> pd.DataFrame:
    """Build a small synthetic daily-stats frame for one creative."""
    rng = np.random.default_rng(0)
    rows = []
    for d in range(n_days):
        rows.append({
            "creative_id": cid,
            "days_since_launch": d,
            "country": "US",
            "os": "ios",
            "impressions": (1000 + d * 50) if with_imps else 0,
            "viewable_impressions": (700 + d * 30) if with_imps else 0,
            "clicks": int(20 + rng.integers(0, 5)) if with_imps else 0,
            "conversions": 2 if with_imps else 0,
            "spend_usd": 10.0 if with_imps else 0.0,
            "revenue_usd": 25.0 if with_imps else 0.0,
            "video_completions": 100 if with_imps else 0,
        })
    return pd.DataFrame(rows)


def test_empty_creative_ids_returns_zero_rows():
    daily = _toy_daily()
    X, names = compute_early_features(daily, creative_ids=[], window=7)
    assert X.shape[0] == 0
    assert X.shape[1] == len(names)


def test_zero_impressions_yields_zero_rates():
    daily = _toy_daily(with_imps=False)
    X, names = compute_early_features(daily, creative_ids=[1], window=7)
    # Find rate columns; they must all be 0 (no impressions, no clicks, etc.)
    for rate in ["early_ctr", "early_cvr", "early_viewability",
                 "early_video_completion_rate", "early_ipm", "early_roas"]:
        idx = names.index(rate)
        assert X[0, idx] == 0.0, f"{rate} should be 0 but was {X[0, idx]}"


def test_output_shape_matches_creative_ids_x_features():
    daily = _toy_daily(cid=1)
    cids = [1, 2, 3]  # 2 and 3 are unknown -> rows zero-filled
    X, names = compute_early_features(daily, creative_ids=cids, window=7)
    assert X.shape == (len(cids), len(names))
    # Unknown creatives -> entirely-zero rows
    assert np.all(X[1] == 0)
    assert np.all(X[2] == 0)
    # Known creative should have nonzero clicks-sum
    assert X[0, names.index("early_clicks_sum")] > 0
