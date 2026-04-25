from typing import Dict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import linregress


class TimeSeriesFeatureExtractor:
    def extract_features(self, ts: pd.DataFrame) -> Dict[str, float]:
        if ts.empty or "days_since_launch" not in ts.columns:
            return {}

        ts = ts.sort_values("days_since_launch").copy()
        ctr = ts["clicks"] / ts["impressions"].replace(0, np.nan)
        ctr = ctr.fillna(0).values
        days = ts["days_since_launch"].values.astype(float)
        n = len(ctr)

        feats: Dict[str, float] = {}

        feats["mean_ctr"] = float(np.mean(ctr))
        feats["std_ctr"] = float(np.std(ctr))
        feats["peak_ctr"] = float(np.max(ctr))
        feats["peak_ctr_day"] = float(days[np.argmax(ctr)]) if n > 0 else 0.0
        feats["min_ctr"] = float(np.min(ctr))
        feats["total_days"] = float(n)

        if n >= 7:
            feats["first_7d_mean_ctr"] = float(np.mean(ctr[:7]))
            feats["last_7d_mean_ctr"] = float(np.mean(ctr[-7:]))
            feats["ctr_decay_ratio"] = (
                feats["last_7d_mean_ctr"] / (feats["first_7d_mean_ctr"] + 1e-8)
            )
        else:
            feats["first_7d_mean_ctr"] = feats["mean_ctr"]
            feats["last_7d_mean_ctr"] = feats["mean_ctr"]
            feats["ctr_decay_ratio"] = 1.0

        if n >= 3:
            slope, _, r, _, _ = linregress(days, ctr)
            feats["ctr_slope"] = float(slope)
            feats["ctr_r2"] = float(r ** 2)
        else:
            feats["ctr_slope"] = 0.0
            feats["ctr_r2"] = 0.0

        if n >= 14:
            slope14, _, _, _, _ = linregress(days[:14], ctr[:14])
            feats["ctr_slope_first14d"] = float(slope14)
        else:
            feats["ctr_slope_first14d"] = feats["ctr_slope"]

        peak = feats["peak_ctr"]
        feats["days_above_50pct_peak"] = float(np.sum(ctr >= 0.5 * peak))
        feats["ctr_drop_from_peak"] = float(peak - ctr[-1]) if n > 0 else 0.0
        feats["pct_drop_from_peak"] = feats["ctr_drop_from_peak"] / (peak + 1e-8)

        # Spend efficiency
        if "spend_usd" in ts.columns and "conversions" in ts.columns:
            spend = ts["spend_usd"].fillna(0).values
            convs = ts["conversions"].fillna(0).values
            total_spend = spend.sum()
            total_convs = convs.sum()
            feats["cost_per_conversion"] = float(total_spend / (total_convs + 1e-8))
            feats["total_spend"] = float(total_spend)
        else:
            feats["cost_per_conversion"] = 0.0
            feats["total_spend"] = 0.0

        # Country/OS diversity
        if "country" in ts.columns:
            feats["country_diversity"] = float(ts["country"].nunique())
        if "os" in ts.columns:
            feats["os_split_android"] = float(
                (ts["os"] == "Android").sum() / len(ts)
            )

        return feats

    def build_fatigue_features(self, ts: pd.DataFrame) -> Dict[str, float]:
        feats = self.extract_features(ts)
        if ts.empty:
            return feats

        ts = ts.sort_values("days_since_launch").copy()
        ctr = ts["clicks"] / ts["impressions"].replace(0, np.nan)
        ctr = ctr.fillna(0).values
        days = ts["days_since_launch"].values.astype(float)
        n = len(ctr)

        # Smoothed curve
        if n >= 7:
            window = min(7, n if n % 2 == 1 else n - 1)
            if window >= 3:
                try:
                    smoothed = savgol_filter(ctr, window_length=window, polyorder=2)
                    diff = np.diff(smoothed)
                    feats["smoothed_ctr_slope_last3"] = float(np.mean(diff[-3:]))
                    feats["max_acceleration"] = float(np.min(diff))
                    feats["elbow_day"] = float(days[np.argmin(diff) + 1]) if len(diff) > 0 else 0.0
                except Exception:
                    feats["smoothed_ctr_slope_last3"] = feats.get("ctr_slope", 0.0)
                    feats["max_acceleration"] = 0.0
                    feats["elbow_day"] = 0.0
        else:
            feats["smoothed_ctr_slope_last3"] = feats.get("ctr_slope", 0.0)
            feats["max_acceleration"] = 0.0
            feats["elbow_day"] = 0.0

        peak = feats.get("peak_ctr", ctr.max() if n > 0 else 0.0)
        current = float(ctr[-1]) if n > 0 else 0.0
        feats["fatigue_score"] = float(max(0.0, (peak - current) / (peak + 1e-8)))

        return feats

    def compute_fatigue_curve(self, ts: pd.DataFrame) -> pd.DataFrame:
        if ts.empty:
            return pd.DataFrame()

        ts = ts.sort_values("days_since_launch").copy()
        ctr = (ts["clicks"] / ts["impressions"].replace(0, np.nan)).fillna(0)
        peak = ctr.max()

        result = ts[["date", "days_since_launch"]].copy()
        result["ctr"] = ctr.values
        result["fatigue_score"] = ((peak - ctr) / (peak + 1e-8)).clip(0, 1).values
        result["rolling_ctr_3d"] = ctr.rolling(3, min_periods=1).mean().values
        return result
