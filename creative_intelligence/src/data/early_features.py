"""
Early-life signal features built from creative_daily_country_os_stats.csv.

These are legitimate predictors: they describe what happened in the first N days
after launch — used to predict full-cycle creative_status / perf_score.
They do NOT leak the label, because the label is generated from the FULL lifecycle
(decay between first-7d and last-7d), and these features only see the first 7 days.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd


def compute_early_features(
    daily_df: pd.DataFrame, creative_ids: List[int], window: int = 7
) -> Tuple[np.ndarray, List[str]]:
    """For each creative_id, build an early-window feature vector.

    Aggregates the first `window` days of daily stats. Returns (n_creatives, n_feats).
    Rows for unknown creative_ids are zero-filled.
    """
    sub = daily_df[daily_df["days_since_launch"] <= window].copy()

    # Per-day totals across all (country, os) splits → one row per (cid, day)
    grouped = sub.groupby(["creative_id", "days_since_launch"]).agg(
        impressions=("impressions", "sum"),
        viewable_impressions=("viewable_impressions", "sum"),
        clicks=("clicks", "sum"),
        conversions=("conversions", "sum"),
        spend_usd=("spend_usd", "sum"),
        revenue_usd=("revenue_usd", "sum"),
        video_completions=("video_completions", "sum"),
    ).reset_index()

    # Per-creative aggregates over the window
    by_cid = grouped.groupby("creative_id")
    agg = by_cid.agg(
        early_impressions_sum=("impressions", "sum"),
        early_clicks_sum=("clicks", "sum"),
        early_conversions_sum=("conversions", "sum"),
        early_spend_sum=("spend_usd", "sum"),
        early_revenue_sum=("revenue_usd", "sum"),
        early_viewable_sum=("viewable_impressions", "sum"),
        early_video_completions_sum=("video_completions", "sum"),
        early_impressions_mean=("impressions", "mean"),
        early_clicks_mean=("clicks", "mean"),
        early_impressions_std=("impressions", "std"),
        early_days_observed=("impressions", "count"),
    ).fillna(0)

    # Derived rates
    agg["early_ctr"] = agg["early_clicks_sum"] / agg["early_impressions_sum"].replace(0, np.nan)
    agg["early_cvr"] = agg["early_conversions_sum"] / agg["early_clicks_sum"].replace(0, np.nan)
    agg["early_viewability"] = agg["early_viewable_sum"] / agg["early_impressions_sum"].replace(0, np.nan)
    agg["early_video_completion_rate"] = (
        agg["early_video_completions_sum"] / agg["early_impressions_sum"].replace(0, np.nan)
    )
    agg["early_ipm"] = 1000 * agg["early_conversions_sum"] / agg["early_impressions_sum"].replace(0, np.nan)
    agg["early_roas"] = agg["early_revenue_sum"] / agg["early_spend_sum"].replace(0, np.nan)
    agg["early_cost_per_click"] = agg["early_spend_sum"] / agg["early_clicks_sum"].replace(0, np.nan)
    agg = agg.fillna(0)

    # Trend: slope of daily CTR within the window (positive = improving)
    def _slope(g: pd.DataFrame) -> float:
        if len(g) < 2:
            return 0.0
        days = g["days_since_launch"].values.astype(np.float32)
        ctr = (g["clicks"] / g["impressions"].replace(0, np.nan)).fillna(0).values.astype(np.float32)
        if np.std(days) == 0:
            return 0.0
        return float(np.polyfit(days, ctr, 1)[0])

    slopes = grouped.groupby("creative_id").apply(_slope, include_groups=False).rename("early_ctr_slope")
    agg = agg.join(slopes).fillna(0)

    # Diversity: how many distinct (country, os) the creative ran on in the window
    diversity = sub.groupby("creative_id").agg(
        early_n_countries=("country", "nunique"),
        early_n_os=("os", "nunique"),
    )
    agg = agg.join(diversity).fillna(0)

    feature_names = list(agg.columns)
    out = np.zeros((len(creative_ids), len(feature_names)), dtype=np.float32)
    cid_to_row = {cid: i for i, cid in enumerate(agg.index.astype(int).tolist())}
    for i, cid in enumerate(creative_ids):
        j = cid_to_row.get(int(cid))
        if j is not None:
            out[i] = agg.iloc[j].values.astype(np.float32)
    return out, feature_names
