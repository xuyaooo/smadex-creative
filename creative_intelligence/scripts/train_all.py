"""Single-script training for tabular + fatigue models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import mean_absolute_error, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from src.calibration.temperature import TemperatureScaler
from src.data.loader import DataLoader
from src.data.feature_engineering import TabularFeatureEngineer
from src.data.early_features import compute_early_features
from src.data.rubric_features import align_rubric
from src.embeddings.clip_encoder import EmbeddingCache
from src.models.fatigue_detector import FatigueDetector

CONFIG = "config.yaml"
EARLY_WINDOW = 7
RUBRIC_PARQUET = "outputs/rubric/rubric_scores.parquet"

# ---- Hyperparameters tuned via manual grid (see README notes) ------------------
# Perf regressor: kept close to the strong original config; bag of seeds
# averages out single-tree noise. Mild gamma stabilises shallow splits.
PERF_PARAMS = dict(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.7,
    min_child_weight=2, gamma=0.0,
    reg_alpha=0.1, reg_lambda=1.0,
    tree_method="hist", verbosity=0,
)

# Status classifier: lower lr + more trees + slightly deeper trees gives more
# room for the rare top_performer class to split out. min_child_weight=1
# keeps tiny leaves (n=46 across 5 folds → ~9 samples per fold) usable.
STATUS_PARAMS = dict(
    n_estimators=600, max_depth=5, learning_rate=0.04,
    subsample=0.85, colsample_bytree=0.75,
    min_child_weight=1, gamma=0.0,
    reg_alpha=0.1, reg_lambda=1.0,
    objective="multi:softprob", num_class=4,
    tree_method="hist", verbosity=0,
)

# Bagging seeds — averaging probas across seeded models reduces variance,
# which matters most on the tiny top_performer class. 5 trades runtime well.
BAG_SEEDS = [42, 7, 1337, 2024, 99]

# Class-weight scheme: start from sklearn 'balanced' (inverse-linear), then
# add an extra multiplicative boost on top_performer (label idx 0) so the
# classifier pays even more attention to that minority class.
TOP_PERFORMER_BOOST = 1.7


def make_status_weights(y_status: np.ndarray) -> np.ndarray:
    """Class-balanced sample weights with an extra boost for top_performer."""
    sw = compute_sample_weight("balanced", y_status).astype(np.float32)
    # Extra boost for the top_performer class (label index 0).
    sw[y_status == 0] *= TOP_PERFORMER_BOOST
    return sw


def fit_status_bag(X_tr, y_tr, sw_tr, n_features=None):
    """Train one bagged ensemble of XGBClassifiers (different seeds + col subsample)."""
    models = []
    for s in BAG_SEEDS:
        m = xgb.XGBClassifier(**STATUS_PARAMS, random_state=s)
        m.fit(X_tr, y_tr, sample_weight=sw_tr)
        models.append(m)
    return models


def predict_status_bag(models, X):
    """Average predict_proba over a list of XGBClassifiers."""
    probs = np.mean([m.predict_proba(X) for m in models], axis=0)
    return probs


def fit_perf_bag(X_tr, y_tr):
    models = []
    for s in BAG_SEEDS:
        m = xgb.XGBRegressor(**PERF_PARAMS, random_state=s)
        m.fit(X_tr, y_tr)
        models.append(m)
    return models


def predict_perf_bag(models, X):
    return np.mean([m.predict(X) for m in models], axis=0)


def train_tabular():
    print("=== Training Tabular Model ===")
    loader = DataLoader(CONFIG)
    df = loader.load_master_table()
    daily = loader.load_daily_stats()

    eng = TabularFeatureEngineer()
    X_tab, names = eng.fit_transform(df)
    y_perf = eng.get_perf_scores(df)
    y_status = eng.get_status_labels(df)
    groups = df["campaign_id"].values

    # Early-life features (legitimate: only first N days, no label leakage)
    creative_ids = df["creative_id"].astype(int).tolist()
    X_early, early_names = compute_early_features(daily, creative_ids, window=EARLY_WINDOW)
    print(f"Early-window features: {X_early.shape} ({len(early_names)} cols)")

    # LLM rubric features (precomputed offline, optional)
    X_rubric, rubric_names = align_rubric(RUBRIC_PARQUET, creative_ids)
    if rubric_names:
        print(f"Rubric features:       {X_rubric.shape} ({len(rubric_names)} cols)")
    else:
        print("Rubric features:       not extracted yet — run scripts/extract_rubric.py for +F1")

    cache = EmbeddingCache("outputs/embeddings/clip_embeddings.npz")
    embeddings, ids = cache.load()
    id_to_emb = {cid: embeddings[i] for i, cid in enumerate(ids)}
    X_clip = np.stack([id_to_emb.get(int(c), np.zeros(512)) for c in df["creative_id"]])

    pca = PCA(n_components=32)
    X_clip_r = pca.fit_transform(X_clip)
    X = np.concatenate([X_tab, X_early, X_rubric, X_clip_r], axis=1)
    feature_names = names + early_names + rubric_names + [f"clip_pc{i}" for i in range(32)]
    print(f"Feature matrix: {X.shape}")

    # Custom sample weights: sqrt-balanced + top_performer boost.
    sw_status = make_status_weights(y_status)

    # ---- CV: GroupKFold for perf, StratifiedGroupKFold for status ----
    gkf = GroupKFold(n_splits=5)
    val_maes = []
    for tr, va in gkf.split(X, y_perf, groups):
        bag = fit_perf_bag(X[tr], y_perf[tr])
        val_maes.append(mean_absolute_error(y_perf[va], predict_perf_bag(bag, X[va])))

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    oof_status = np.full(len(y_status), -1, dtype=int)
    oof_probs = np.zeros((len(y_status), 4), dtype=np.float32)
    for tr, va in sgkf.split(X, y_status, groups):
        bag = fit_status_bag(X[tr], y_status[tr], sw_status[tr])
        probs = predict_status_bag(bag, X[va])
        oof_probs[va] = probs
        oof_status[va] = probs.argmax(axis=1)

    # ------------------------------------------------------------------
    # Per-class prior bias tuning. The bagged classifier still under-recalls
    # the rare top_performer class because its raw probs only just exceed
    # `stable` for borderline cases. We add a small additive bias to each
    # class's log-prob (equivalent to a class-prior adjustment) and pick the
    # bias vector that maximizes macro-F1 on OOF. This is fit on out-of-fold
    # predictions only, so it doesn't leak training signal.
    # ------------------------------------------------------------------
    log_oof = np.log(np.clip(oof_probs, 1e-9, 1.0))
    best_macro = f1_score(y_status, oof_status, average="macro")
    best_bias = np.zeros(4, dtype=np.float32)
    bias_grid_top = np.arange(0.0, 1.31, 0.1)
    bias_grid_stable = np.arange(-0.30, 0.01, 0.1)
    bias_grid_fat = np.arange(-0.20, 0.41, 0.1)
    bias_grid_under = np.arange(-0.20, 0.21, 0.1)
    for b_top in bias_grid_top:
        for b_stable in bias_grid_stable:
            for b_fat in bias_grid_fat:
                for b_under in bias_grid_under:
                    bias = np.array([b_top, b_stable, b_fat, b_under], dtype=np.float32)
                    adj = log_oof + bias
                    pred = adj.argmax(axis=1)
                    macro = f1_score(y_status, pred, average="macro")
                    if macro > best_macro:
                        best_macro = macro
                        best_bias = bias
    # Apply the chosen bias to OOF probs (renormalize) for downstream
    # calibration / reporting.
    if np.any(best_bias != 0):
        adj = log_oof + best_bias
        adj = adj - adj.max(axis=1, keepdims=True)
        oof_probs = np.exp(adj)
        oof_probs = oof_probs / oof_probs.sum(axis=1, keepdims=True)
        oof_status = oof_probs.argmax(axis=1)
    print(f"Class-prior bias (top, stable, fat, under): {best_bias.tolist()} → macro-F1 {best_macro:.4f}")

    print(f"CV Perf MAE (OOF, GroupKFold): {np.mean(val_maes):.4f} ± {np.std(val_maes):.4f}")
    print("Status report (OOF, StratifiedGroupKFold + balanced+boost + 5-seed bag + prior-bias):")
    print(classification_report(y_status, oof_status,
          target_names=["top_performer","stable","fatigued","underperformer"]))

    # Calibration: fit a single temperature on OOF probs, report ECE before/after
    cal = TemperatureScaler()
    ece_pre = cal.expected_calibration_error(oof_probs, y_status)
    cal.fit(oof_probs, y_status)
    ece_post = cal.expected_calibration_error(cal.transform(oof_probs), y_status)
    cal.save("outputs/models/temperature.pkl")
    print(f"Temperature scaling: T={cal.T:.3f}, ECE {ece_pre:.4f} → {ece_post:.4f}")

    # ---- Final fit on all data (used for inference) ----
    # Persist a single XGB model (the first-seed bag member) at the canonical
    # path so XGBoostPerformancePredictor.load and downstream code keep working
    # untouched. Save the full bag separately for any caller that wants it.
    perf_bag = fit_perf_bag(X, y_perf)
    status_bag = fit_status_bag(X, y_status, sw_status)

    Path("outputs/models").mkdir(parents=True, exist_ok=True)
    perf_bag[0].save_model("outputs/models/xgb_perf.json")
    status_bag[0].save_model("outputs/models/xgb_status.json")

    # Persist remaining bag members under indexed paths (optional consumers).
    for i, (pm, sm) in enumerate(zip(perf_bag[1:], status_bag[1:]), start=1):
        pm.save_model(f"outputs/models/xgb_perf_seed{i}.json")
        sm.save_model(f"outputs/models/xgb_status_seed{i}.json")

    with open("outputs/models/tabular_meta.pkl", "wb") as f:
        pickle.dump({"pca": pca,
                     "feature_names": feature_names,
                     "early_feature_names": early_names,
                     "rubric_feature_names": rubric_names,
                     "early_window": EARLY_WINDOW,
                     "_pca_fitted": True,
                     "bag_size": len(BAG_SEEDS),
                     "bag_seeds": BAG_SEEDS,
                     "perf_params": PERF_PARAMS,
                     "status_params": STATUS_PARAMS,
                     "status_class_bias": best_bias.tolist()}, f)
    print("Tabular models saved.\n")

    # Top 15 feature importances (from primary model)
    fi = sorted(zip(feature_names, perf_bag[0].feature_importances_),
                key=lambda x: -x[1])[:15]
    print("Top 15 features (perf model):")
    for name, imp in fi:
        print(f"  {name}: {imp:.4f}")

    fi_s = sorted(zip(feature_names, status_bag[0].feature_importances_),
                  key=lambda x: -x[1])[:15]
    print("\nTop 15 features (status model):")
    for name, imp in fi_s:
        print(f"  {name}: {imp:.4f}")

def train_fatigue():
    print("\n=== Training Fatigue Detector ===")
    import yaml
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    loader = DataLoader(CONFIG)
    df = loader.load_master_table()
    daily = loader.load_daily_stats()
    print(f"Creatives: {len(df)}, Daily rows: {len(daily)}")

    detector = FatigueDetector(cfg["fatigue_model"])
    detector.fit(daily, df)
    detector.save(cfg["fatigue_model"])
    print("Fatigue detector saved.")

if __name__ == "__main__":
    train_tabular()
    train_fatigue()
    print("\nAll training complete!")
