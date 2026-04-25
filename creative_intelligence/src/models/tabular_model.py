import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

STATUS_LABELS = ["top_performer", "stable", "fatigued", "underperformer"]


class XGBoostPerformancePredictor:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.perf_model = xgb.XGBRegressor(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 5),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.7),
            tree_method="hist",
            random_state=42,
        )
        self.status_model = xgb.XGBClassifier(
            n_estimators=cfg.get("n_estimators", 300),
            max_depth=cfg.get("max_depth", 5),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.7),
            objective="multi:softprob",
            num_class=4,
            tree_method="hist",
            random_state=42,
        )
        self.pca = PCA(n_components=cfg.get("pca_components", 32))
        self.feature_names: List[str] = []
        self._pca_fitted = False

    def _merge_features(self, X_tab: np.ndarray, X_clip: np.ndarray, fit_pca: bool = False) -> np.ndarray:
        if fit_pca:
            clip_reduced = self.pca.fit_transform(X_clip)
            self._pca_fitted = True
        else:
            clip_reduced = self.pca.transform(X_clip) if self._pca_fitted else X_clip
        return np.concatenate([X_tab, clip_reduced], axis=1)

    def fit(
        self,
        X_tab: np.ndarray,
        X_clip: np.ndarray,
        y_perf: np.ndarray,
        y_status: np.ndarray,
        groups: np.ndarray,
        feature_names: List[str],
    ) -> Dict[str, float]:
        self.feature_names = feature_names + [f"clip_pc{i}" for i in range(self.pca.n_components)]
        X = self._merge_features(X_tab, X_clip, fit_pca=True)

        # Cross-validation
        gkf = GroupKFold(n_splits=self.cfg.get("cv_splits", 5))
        perf_maes, status_f1s = [], []
        for train_idx, val_idx in gkf.split(X, y_perf, groups):
            self.perf_model.fit(X[train_idx], y_perf[train_idx])
            self.status_model.fit(X[train_idx], y_status[train_idx])
            perf_maes.append(mean_absolute_error(y_perf[val_idx], self.perf_model.predict(X[val_idx])))

        # Final fit on all data
        self.perf_model.fit(X, y_perf)
        self.status_model.fit(X, y_status)

        return {"perf_mae_cv": float(np.mean(perf_maes))}

    def predict_perf_score(self, X_tab: np.ndarray, X_clip: np.ndarray) -> np.ndarray:
        X = self._merge_features(X_tab, X_clip)
        return self.perf_model.predict(X).astype(np.float32)

    def predict_status(self, X_tab: np.ndarray, X_clip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = self._merge_features(X_tab, X_clip)
        probs = self.status_model.predict_proba(X)
        labels = probs.argmax(axis=1)
        return labels, probs

    def explain_prediction(self, X_tab: np.ndarray, X_clip: np.ndarray) -> Dict[str, float]:
        # Use XGBoost's native pred_contribs (SHAP values without the shap library)
        X = self._merge_features(X_tab, X_clip)
        dmat = xgb.DMatrix(X)
        contribs = self.perf_model.get_booster().predict(dmat, pred_contribs=True)
        # contribs shape: (n_samples, n_features + 1) — last col is bias
        feature_contribs = contribs[0, :-1]
        return dict(zip(self.feature_names, feature_contribs.tolist()))

    def get_feature_importances(self) -> pd.DataFrame:
        importances = self.perf_model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names[:len(importances)],
            "importance": importances,
        }).sort_values("importance", ascending=False)

    def save(self, cfg: dict) -> None:
        root = Path(cfg.get("perf_model_path", "outputs/models/xgb_perf.json")).parent
        root.mkdir(parents=True, exist_ok=True)
        self.perf_model.save_model(cfg.get("perf_model_path", "outputs/models/xgb_perf.json"))
        self.status_model.save_model(cfg.get("status_model_path", "outputs/models/xgb_status.json"))
        with open(root / "tabular_meta.pkl", "wb") as f:
            pickle.dump({"pca": self.pca, "feature_names": self.feature_names,
                         "_pca_fitted": self._pca_fitted}, f)

    @classmethod
    def load(cls, cfg: dict) -> "XGBoostPerformancePredictor":
        obj = cls(cfg)
        obj.perf_model.load_model(cfg.get("perf_model_path", "outputs/models/xgb_perf.json"))
        obj.status_model.load_model(cfg.get("status_model_path", "outputs/models/xgb_status.json"))
        root = Path(cfg.get("perf_model_path", "outputs/models/xgb_perf.json")).parent
        with open(root / "tabular_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        obj.pca = meta["pca"]
        obj.feature_names = meta["feature_names"]
        obj._pca_fitted = meta["_pca_fitted"]
        return obj
