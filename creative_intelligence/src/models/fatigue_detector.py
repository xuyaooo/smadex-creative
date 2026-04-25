import pickle
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.time_series_features import TimeSeriesFeatureExtractor


class FatigueDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.early_window = cfg.get("early_window_days", 7)
        self.threshold = cfg.get("fatigue_score_threshold", 0.5)
        self.ts_extractor = TimeSeriesFeatureExtractor()
        self.classifier = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=42
        )
        self.regressor = lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=42
        )
        self._feature_names: List[str] = []

    def _build_features(self, daily_df: pd.DataFrame, creative_ids: List[int], window_days: int) -> np.ndarray:
        rows = []
        for cid in creative_ids:
            ts = daily_df[daily_df["creative_id"] == cid].sort_values("days_since_launch")
            ts_window = ts[ts["days_since_launch"] <= window_days]
            feats = self.ts_extractor.build_fatigue_features(ts_window)
            rows.append(feats)

        if not rows:
            return np.zeros((0, 1))

        df = pd.DataFrame(rows).fillna(0)
        self._feature_names = df.columns.tolist()
        return df.values.astype(np.float32)

    def fit(self, daily_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
        creative_ids = summary_df["creative_id"].tolist()
        y_fatigued = (summary_df["creative_status"] == "fatigued").astype(int).values
        y_day = summary_df["fatigue_day"].fillna(0).values.astype(np.float32)

        X = self._build_features(daily_df, creative_ids, self.early_window)
        if X.shape[0] == 0:
            return

        self.classifier.fit(X, y_fatigued)

        fatigued_mask = y_fatigued == 1
        if fatigued_mask.sum() >= 5:
            self.regressor.fit(X[fatigued_mask], y_day[fatigued_mask])

    def predict_fatigue_risk(self, creative_id: int, daily_df: pd.DataFrame) -> Dict:
        ts = daily_df[daily_df["creative_id"] == creative_id].sort_values("days_since_launch")
        ts_window = ts[ts["days_since_launch"] <= self.early_window]
        feats = self.ts_extractor.build_fatigue_features(ts_window)
        X = pd.DataFrame([feats]).reindex(columns=self._feature_names, fill_value=0).values.astype(np.float32)

        will_fatigue = bool(self.classifier.predict(X)[0])
        fatigue_prob = float(self.classifier.predict_proba(X)[0][1])
        fatigue_day_est = int(self.regressor.predict(X)[0]) if will_fatigue else -1

        current_feats = self.ts_extractor.build_fatigue_features(ts)
        current_score = float(current_feats.get("fatigue_score", 0.0))
        days_so_far = int(ts["days_since_launch"].max()) if not ts.empty else 0

        return {
            "will_fatigue": will_fatigue,
            "fatigue_probability": fatigue_prob,
            "fatigue_day_estimate": fatigue_day_est,
            "current_fatigue_score": current_score,
            "days_active": days_so_far,
            "days_remaining_estimate": max(0, fatigue_day_est - days_so_far) if will_fatigue else -1,
        }

    def get_fatigue_signals(self, creative_id: int, daily_df: pd.DataFrame) -> List[str]:
        ts = daily_df[daily_df["creative_id"] == creative_id].sort_values("days_since_launch")
        feats = self.ts_extractor.build_fatigue_features(ts)
        signals = []
        if feats.get("pct_drop_from_peak", 0) > 0.4:
            signals.append(f"CTR dropped {feats['pct_drop_from_peak']:.0%} from peak")
        if feats.get("ctr_slope", 0) < -0.0005:
            signals.append("Negative CTR trend (linear slope < 0)")
        if feats.get("days_above_50pct_peak", 0) < 5:
            signals.append("Creative stayed above 50% peak CTR for < 5 days")
        if feats.get("fatigue_score", 0) > self.threshold:
            signals.append(f"Fatigue score {feats['fatigue_score']:.2f} exceeds threshold {self.threshold}")
        return signals if signals else ["No significant fatigue signals detected"]

    def compute_fatigue_curve(self, creative_id: int, daily_df: pd.DataFrame) -> pd.DataFrame:
        ts = daily_df[daily_df["creative_id"] == creative_id]
        return self.ts_extractor.compute_fatigue_curve(ts)

    def save(self, cfg: dict) -> None:
        clf_path = Path(cfg.get("classifier_path", "outputs/models/fatigue_clf.pkl"))
        reg_path = Path(cfg.get("regressor_path", "outputs/models/fatigue_reg.pkl"))
        clf_path.parent.mkdir(parents=True, exist_ok=True)
        with open(clf_path, "wb") as f:
            pickle.dump({"model": self.classifier, "feature_names": self._feature_names}, f)
        with open(reg_path, "wb") as f:
            pickle.dump({"model": self.regressor}, f)

    @classmethod
    def load(cls, cfg: dict) -> "FatigueDetector":
        obj = cls(cfg)
        clf_path = cfg.get("classifier_path", "outputs/models/fatigue_clf.pkl")
        reg_path = cfg.get("regressor_path", "outputs/models/fatigue_reg.pkl")
        with open(clf_path, "rb") as f:
            data = pickle.load(f)
            obj.classifier = data["model"]
            obj._feature_names = data["feature_names"]
        with open(reg_path, "rb") as f:
            obj.regressor = pickle.load(f)["model"]
        return obj
