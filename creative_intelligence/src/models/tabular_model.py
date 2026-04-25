import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

STATUS_LABELS = ["top_performer", "stable", "fatigued", "underperformer"]

# Tuned defaults — kept in sync with scripts/train_all.py.
# Lower learning rate + more trees + slightly deeper trees + a touch of gamma
# improved CV MAE for the regressor and lifted top_performer recall on the
# status classifier without hurting the other classes.
DEFAULT_PERF_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=2, gamma=0.0,
    reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist",
)
DEFAULT_STATUS_PARAMS = dict(
    n_estimators=600, max_depth=5, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.75,
    min_child_weight=1, gamma=0.0,
    reg_alpha=0.1, reg_lambda=1.0,
    objective="multi:softprob", num_class=4,
    tree_method="hist",
)


def _xgb_regressor(cfg: dict, seed: int = 42) -> xgb.XGBRegressor:
    """Build an XGBRegressor, allowing the caller to override any default."""
    params = dict(DEFAULT_PERF_PARAMS)
    # Allow the legacy short-name overrides for backwards compatibility.
    for k in ("n_estimators", "max_depth", "learning_rate",
              "subsample", "colsample_bytree",
              "min_child_weight", "gamma",
              "reg_alpha", "reg_lambda"):
        if k in cfg:
            params[k] = cfg[k]
    return xgb.XGBRegressor(random_state=seed, verbosity=0, **params)


def _xgb_classifier(cfg: dict, seed: int = 42) -> xgb.XGBClassifier:
    params = dict(DEFAULT_STATUS_PARAMS)
    for k in ("n_estimators", "max_depth", "learning_rate",
              "subsample", "colsample_bytree",
              "min_child_weight", "gamma",
              "reg_alpha", "reg_lambda"):
        if k in cfg:
            params[k] = cfg[k]
    return xgb.XGBClassifier(random_state=seed, verbosity=0, **params)


class XGBoostPerformancePredictor:
    """
    XGBoost-based regressor + classifier for creative-performance scoring and
    status labelling.

    The persistence format is unchanged: a primary regressor at
    ``outputs/models/xgb_perf.json`` and a primary classifier at
    ``outputs/models/xgb_status.json``, with metadata in ``tabular_meta.pkl``.

    If additional bag-mate JSON files exist alongside the primary models
    (``xgb_perf_seed{i}.json`` / ``xgb_status_seed{i}.json``), they are loaded
    and used as a small averaging ensemble for predictions — this gives a
    meaningful boost on the small ``top_performer`` class. Inference falls
    back to the single primary model if no bag-mates are present, which keeps
    older artifacts loadable.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.perf_model = _xgb_regressor(cfg)
        self.status_model = _xgb_classifier(cfg)
        self.pca = PCA(n_components=cfg.get("pca_components", 32))
        self.feature_names: List[str] = []
        self._pca_fitted = False
        # Optional ensemble members loaded from disk.
        self._perf_bag: List[xgb.XGBRegressor] = []
        self._status_bag: List[xgb.XGBClassifier] = []
        # Optional per-class prior bias on log-probs (set by training script
        # via OOF macro-F1 maximisation; absent → no shift).
        self._status_class_bias: Optional[np.ndarray] = None

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
        sample_weight_status: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        self.feature_names = feature_names + [f"clip_pc{i}" for i in range(self.pca.n_components)]
        X = self._merge_features(X_tab, X_clip, fit_pca=True)

        # Cross-validation MAE (GroupKFold) for the perf model — informational.
        gkf = GroupKFold(n_splits=self.cfg.get("cv_splits", 5))
        perf_maes: List[float] = []
        for train_idx, val_idx in gkf.split(X, y_perf, groups):
            self.perf_model.fit(X[train_idx], y_perf[train_idx])
            sw = None if sample_weight_status is None else sample_weight_status[train_idx]
            self.status_model.fit(X[train_idx], y_status[train_idx], sample_weight=sw)
            perf_maes.append(mean_absolute_error(y_perf[val_idx], self.perf_model.predict(X[val_idx])))

        # Final fit on all data
        self.perf_model.fit(X, y_perf)
        self.status_model.fit(X, y_status, sample_weight=sample_weight_status)

        return {"perf_mae_cv": float(np.mean(perf_maes))}

    # ---------------- Inference ----------------
    def _perf_predict(self, X: np.ndarray) -> np.ndarray:
        if self._perf_bag:
            preds = [self.perf_model.predict(X)] + [m.predict(X) for m in self._perf_bag]
            return np.mean(preds, axis=0).astype(np.float32)
        return self.perf_model.predict(X).astype(np.float32)

    def _status_predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._status_bag:
            probs = [self.status_model.predict_proba(X)] + [m.predict_proba(X) for m in self._status_bag]
            probs = np.mean(probs, axis=0)
        else:
            probs = self.status_model.predict_proba(X)
        if self._status_class_bias is not None and np.any(self._status_class_bias != 0):
            log_p = np.log(np.clip(probs, 1e-9, 1.0)) + self._status_class_bias
            log_p = log_p - log_p.max(axis=1, keepdims=True)
            probs = np.exp(log_p)
            probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict_perf_score(self, X_tab: np.ndarray, X_clip: np.ndarray) -> np.ndarray:
        X = self._merge_features(X_tab, X_clip)
        return self._perf_predict(X)

    def predict_status(self, X_tab: np.ndarray, X_clip: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = self._merge_features(X_tab, X_clip)
        probs = self._status_predict_proba(X)
        labels = probs.argmax(axis=1)
        return labels, probs

    def explain_prediction(self, X_tab: np.ndarray, X_clip: np.ndarray) -> Dict[str, float]:
        # Use XGBoost's native pred_contribs (SHAP values without the shap library).
        # Explanations always come from the primary model — bag-mates are a noise-
        # reduction trick, but a single tree's contribs is what users want to see.
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

    # ---------------- Persistence ----------------
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
        perf_path = Path(cfg.get("perf_model_path", "outputs/models/xgb_perf.json"))
        status_path = Path(cfg.get("status_model_path", "outputs/models/xgb_status.json"))
        obj.perf_model.load_model(str(perf_path))
        obj.status_model.load_model(str(status_path))

        # Auto-load any bag-mates written by scripts/train_all.py
        # (xgb_perf_seed1.json … xgb_perf_seedN.json).
        root = perf_path.parent
        for i in range(1, 16):
            p_path = root / f"xgb_perf_seed{i}.json"
            s_path = root / f"xgb_status_seed{i}.json"
            if p_path.exists() and s_path.exists():
                pm = _xgb_regressor(cfg)
                pm.load_model(str(p_path))
                sm = _xgb_classifier(cfg)
                sm.load_model(str(s_path))
                obj._perf_bag.append(pm)
                obj._status_bag.append(sm)
            else:
                break

        with open(root / "tabular_meta.pkl", "rb") as f:
            meta = pickle.load(f)
        obj.pca = meta["pca"]
        obj.feature_names = meta["feature_names"]
        obj._pca_fitted = meta["_pca_fitted"]
        bias = meta.get("status_class_bias")
        if bias is not None:
            obj._status_class_bias = np.asarray(bias, dtype=np.float32)
        return obj
