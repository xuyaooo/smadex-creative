"""Production trainer using the cleaned, leakage-free splits.

This is the script that mirrors `notebooks/models.ipynb`:
  * loads from `outputs/splits/{train,val,test}.parquet`
  * trains a 5-model ensemble (XGB-bag + LightGBM + CatBoost + HistGBM + LogReg)
  * fits temperature scaling on val
  * trains the 4-bucket fatigue classifier (Never / Late / Standard / Early)
  * saves all artifacts under `outputs/models/clean/` and writes a metrics
    JSON to `outputs/models/clean_metrics.json`

Backend-compatible: also writes `outputs/models/xgb_status.json` /
`xgb_perf.json` / `temperature.pkl` / `fatigue_clf.pkl` so the FastAPI
server in `backend/main.py` keeps working without changes.

Usage:
    python3 scripts/train_clean.py
    python3 scripts/train_clean.py --no-catboost   # skip catboost (faster)
"""
import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, log_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scipy.optimize import minimize_scalar

import xgboost as xgb
import lightgbm as lgb


REPO = Path(__file__).resolve().parent.parent
SPLITS = REPO / "outputs/splits"
MODELS = REPO / "outputs/models"
CLEAN = MODELS / "clean"

NON_FEATURES = [
    "creative_id", "campaign_id", "creative_status",
    "sample_weight", "cluster", "creative_launch_date", "strat_key",
]

BAG_SEEDS = [42, 1, 2, 3, 4]
XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.04,
    min_child_weight=2, subsample=0.85, colsample_bytree=0.75,
    tree_method="hist", eval_metric="mlogloss", verbosity=0,
)


# ------------------------------------------------------------------ utils

def encode_X(df: pd.DataFrame, feats, encoders):
    X = df[feats].copy()
    for c, le in encoders.items():
        X[c] = le.transform(df[c].astype(str))
    for c in feats:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(np.float32)
    return X


def cb_input(df: pd.DataFrame, feats, cat_cols, num_cols):
    out = df[feats].copy()
    for c in cat_cols:
        out[c] = df[c].astype(str)
    for c in num_cols:
        out[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    return out


def temp_scale(proba: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-9
    z = np.log(np.clip(proba, eps, 1 - eps)) / T
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(proba_val, y_val, n_classes):
    def nll(T):
        return log_loss(y_val, temp_scale(proba_val, T), labels=range(n_classes))

    return float(minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded").x)


def expected_calibration_error(proba, y, n_bins=10) -> float:
    conf, pred = proba.max(1), proba.argmax(1)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / len(y)) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


# ------------------------------------------------------------------ trainers

def train_xgb_bag(X_tr, y_tr, w_tr):
    models = [
        xgb.XGBClassifier(**XGB_PARAMS, random_state=s).fit(X_tr, y_tr, sample_weight=w_tr)
        for s in BAG_SEEDS
    ]
    return models


def xgb_bag_proba(models, X):
    return np.mean([m.predict_proba(X) for m in models], axis=0)


def train_lgb(X_tr, y_tr, w_tr):
    m = lgb.LGBMClassifier(
        n_estimators=400, num_leaves=23, learning_rate=0.04,
        min_child_samples=5, subsample=0.85, colsample_bytree=0.75,
        random_state=42, verbose=-1,
    )
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m


def train_catboost(df_tr, y_tr, w_tr, cat_cols, feats):
    from catboost import CatBoostClassifier

    cat_idx = [feats.index(c) for c in cat_cols]
    m = CatBoostClassifier(
        iterations=500, depth=5, learning_rate=0.05,
        cat_features=cat_idx, random_seed=42, verbose=False,
    )
    m.fit(df_tr, y_tr, sample_weight=w_tr)
    return m


def train_hgb(X_tr, y_tr, w_tr):
    m = HistGradientBoostingClassifier(
        max_iter=400, max_depth=4, learning_rate=0.05,
        class_weight="balanced", random_state=42,
    )
    m.fit(X_tr, y_tr, sample_weight=w_tr)
    return m


def train_logreg(df_tr, y_tr, w_tr, cat_cols, num_cols, feats):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", StandardScaler(), num_cols),
    ])
    pipe = Pipeline([
        ("pre", pre),
        ("lr", LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced",
                                   random_state=42)),
    ])
    pipe.fit(df_tr[feats], y_tr, lr__sample_weight=w_tr)
    return pipe


# ------------------------------------------------------------------ fatigue

def fatigue_bucket(d):
    if pd.isna(d):
        return "never"
    if d >= 14:
        return "late"
    if d >= 12:
        return "standard"
    return "early"


def train_fatigue(train, val, test, X_tr, X_va, X_te):
    raw = pd.read_csv(REPO.parent / "creative_summary.csv",
                      usecols=["creative_id", "fatigue_day"])
    raw["fatigue_bucket"] = raw["fatigue_day"].apply(fatigue_bucket)

    fy_enc = LabelEncoder().fit(["never", "late", "standard", "early"])

    def merge_y(df):
        merged = df.merge(raw[["creative_id", "fatigue_bucket"]],
                          on="creative_id", how="left")
        return fy_enc.transform(merged["fatigue_bucket"])

    y_f_tr, y_f_va, y_f_te = merge_y(train), merge_y(val), merge_y(test)

    fat = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=15,
        min_child_samples=8, class_weight="balanced",
        random_state=42, verbose=-1,
    )
    fat.fit(X_tr, y_f_tr)
    f1_va = f1_score(y_f_va, fat.predict(X_va), average="macro")
    f1_te = f1_score(y_f_te, fat.predict(X_te), average="macro")
    return fat, fy_enc, f1_va, f1_te


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-catboost", action="store_true",
                        help="Skip CatBoost (the slowest model; useful for smoke runs)")
    parser.add_argument("--final", action="store_true",
                        help="Production mode: refit on train ∪ val (hyperparams "
                             "and temperature locked from val-tuned run). Saves to "
                             "outputs/models/final/.")
    args = parser.parse_args()

    print("loading clean splits...")
    train = pd.read_parquet(SPLITS / "train.parquet")
    val   = pd.read_parquet(SPLITS / "val.parquet")
    test  = pd.read_parquet(SPLITS / "test.parquet")
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")

    # In --final mode, combine train + val into one training set (n=860).
    # Hyperparameters and temperature were locked on val in the prior run.
    locked_T = None
    if args.final:
        prior_meta_path = CLEAN / "meta.pkl"
        if not prior_meta_path.exists():
            sys.exit("--final requires a prior val-tuned run. Run "
                     "`python3 scripts/train_clean.py` first to lock hyperparams.")
        with open(prior_meta_path, "rb") as f:
            locked_T = pickle.load(f)["temperature"]
        train = pd.concat([train, val], ignore_index=True)
        # Recompute sample weights on the bigger set so the class-weight
        # scheme stays correct.
        from sklearn.utils.class_weight import compute_sample_weight as _csw
        w = _csw("balanced", train["creative_status"].values).astype(np.float32)
        w[train["creative_status"].values == "top_performer"] *= 1.7
        train["sample_weight"] = w
        print(f"  --final: train ∪ val → {len(train)}; locked T={locked_T:.3f}")

    feats = [c for c in train.columns if c not in NON_FEATURES]
    cat_cols = [c for c in feats if not pd.api.types.is_numeric_dtype(train[c])]
    num_cols = [c for c in feats if c not in cat_cols]
    print(f"  features={len(feats)} ({len(cat_cols)} cat + {len(num_cols)} num)")

    all_rows = pd.concat([train, val, test], ignore_index=True)
    encoders = {c: LabelEncoder().fit(all_rows[c].astype(str)) for c in cat_cols}
    y_enc = LabelEncoder().fit(train["creative_status"])
    class_names = list(y_enc.classes_)

    X_tr = encode_X(train, feats, encoders)
    X_va = encode_X(val, feats, encoders)
    X_te = encode_X(test, feats, encoders)
    y_tr = y_enc.transform(train["creative_status"])
    y_va = y_enc.transform(val["creative_status"])
    y_te = y_enc.transform(test["creative_status"])
    w_tr = train["sample_weight"].values

    # Train models
    timings = {}
    t = time.perf_counter()
    xgb_models = train_xgb_bag(X_tr, y_tr, w_tr)
    timings["xgb_bag_5"] = round(time.perf_counter() - t, 2)
    print(f"  XGB 5-seed bag : {timings['xgb_bag_5']:.2f}s")

    t = time.perf_counter()
    lgb_model = train_lgb(X_tr, y_tr, w_tr)
    timings["lgb"] = round(time.perf_counter() - t, 2)
    print(f"  LightGBM       : {timings['lgb']:.2f}s")

    cb_model = None
    if not args.no_catboost:
        t = time.perf_counter()
        df_tr_cb = cb_input(train, feats, cat_cols, num_cols)
        cb_model = train_catboost(df_tr_cb, y_tr, w_tr, cat_cols, feats)
        timings["catboost"] = round(time.perf_counter() - t, 2)
        print(f"  CatBoost       : {timings['catboost']:.2f}s")

    t = time.perf_counter()
    hgb_model = train_hgb(X_tr, y_tr, w_tr)
    timings["hgb"] = round(time.perf_counter() - t, 2)
    print(f"  HistGBM        : {timings['hgb']:.2f}s")

    t = time.perf_counter()
    lr_model = train_logreg(train, y_tr, w_tr, cat_cols, num_cols, feats)
    timings["logreg"] = round(time.perf_counter() - t, 2)
    print(f"  LogReg         : {timings['logreg']:.2f}s")

    # Per-model val + test probabilities
    df_va_cb = cb_input(val, feats, cat_cols, num_cols) if cb_model else None
    df_te_cb = cb_input(test, feats, cat_cols, num_cols) if cb_model else None

    proba = {
        "xgb_bag":  (xgb_bag_proba(xgb_models, X_va), xgb_bag_proba(xgb_models, X_te)),
        "lgb":      (lgb_model.predict_proba(X_va), lgb_model.predict_proba(X_te)),
        "hgb":      (hgb_model.predict_proba(X_va), hgb_model.predict_proba(X_te)),
        "logreg":   (lr_model.predict_proba(val[feats]), lr_model.predict_proba(test[feats])),
    }
    if cb_model:
        proba["catboost"] = (cb_model.predict_proba(df_va_cb),
                              cb_model.predict_proba(df_te_cb))

    # Ensemble = mean of all available
    p_ens_va = np.mean([p[0] for p in proba.values()], axis=0)
    p_ens_te = np.mean([p[1] for p in proba.values()], axis=0)

    # Per-model val macro-F1 — only meaningful when val is held out.
    val_f1 = {}
    if not args.final:
        print("\nVal macro-F1 by model:")
        for name, (p_va, _) in proba.items():
            f = f1_score(y_va, p_va.argmax(1), average="macro")
            val_f1[name] = round(float(f), 4)
            print(f"  {name:<10} {f:.4f}")
        val_f1["ensemble"] = round(float(f1_score(y_va, p_ens_va.argmax(1), average="macro")), 4)
        print(f"  {'ensemble':<10} {val_f1['ensemble']:.4f}")
    else:
        print("\n(val is now part of training; per-model val F1 not reported "
              "to avoid confusion with held-out scores)")

    # Calibration: in --final mode use the previously-locked T (val is now
    # part of training so we cannot re-tune it without leakage). In standard
    # mode, fit T on the held-out val proba.
    if args.final:
        T_opt = locked_T
        print(f"\nlocked T = {T_opt:.3f} (carried over from val-tuned run)")
    else:
        T_opt = fit_temperature(p_ens_va, y_va, len(class_names))
        p_ens_va_cal = temp_scale(p_ens_va, T_opt)
        print(f"\noptimal T = {T_opt:.3f}")
        print(f"ECE  uncal: {expected_calibration_error(p_ens_va, y_va):.4f}   "
              f"cal: {expected_calibration_error(p_ens_va_cal, y_va):.4f}")
    p_ens_te_cal = temp_scale(p_ens_te, T_opt)

    # Test eval (touch ONCE)
    pred_te = p_ens_te_cal.argmax(1)
    test_f1_macro = float(f1_score(y_te, pred_te, average="macro"))
    test_f1_weighted = float(f1_score(y_te, pred_te, average="weighted"))
    test_acc = float((pred_te == y_te).mean())
    test_ll = float(log_loss(y_te, np.clip(p_ens_te_cal, 1e-9, 1 - 1e-9),
                              labels=range(len(class_names))))
    test_ece = expected_calibration_error(p_ens_te_cal, y_te)
    print(f"\nTEST: macro-F1={test_f1_macro:.4f}  weighted-F1={test_f1_weighted:.4f}  "
          f"acc={test_acc:.4f}  log-loss={test_ll:.4f}  ECE={test_ece:.4f}")
    print(classification_report(y_te, pred_te, target_names=class_names, zero_division=0))

    # Fatigue 4-bucket model
    fat, fy_enc, fat_f1_va, fat_f1_te = train_fatigue(train, val, test, X_tr, X_va, X_te)
    if args.final:
        print(f"\nFatigue 4-bucket   test macro-F1={fat_f1_te:.3f}  "
              f"(val merged into train; no held-out val score)")
    else:
        print(f"\nFatigue 4-bucket   val macro-F1={fat_f1_va:.3f}   test={fat_f1_te:.3f}")

    # Persist — separate folder for --final so val-tuned artifacts stay around
    out_dir = MODELS / ("final" if args.final else "clean")
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # Save individual models
    for i, m in enumerate(xgb_models):
        m.save_model(str(out_dir / f"xgb_seed{i}.json"))
    with open(out_dir / "lgb.pkl", "wb") as f:
        pickle.dump(lgb_model, f)
    if cb_model:
        cb_model.save_model(str(out_dir / "catboost.cbm"))
    with open(out_dir / "hgb.pkl", "wb") as f:
        pickle.dump(hgb_model, f)
    with open(out_dir / "logreg.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    with open(out_dir / "fatigue_4bucket.pkl", "wb") as f:
        pickle.dump({"model": fat, "encoder": fy_enc}, f)

    # Save metadata: encoders, feature lists, label encoder, temperature
    meta = {
        "feats": feats,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "encoders": encoders,
        "y_encoder": y_enc,
        "class_names": class_names,
        "temperature": T_opt,
        "bag_seeds": BAG_SEEDS,
        "use_catboost": cb_model is not None,
    }
    with open(out_dir / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    # Backend-compat shims
    xgb_models[0].save_model(str(MODELS / "xgb_status.json"))
    with open(MODELS / "tabular_meta.pkl", "wb") as f:
        pickle.dump({"feats": feats, "cat_cols": cat_cols,
                      "encoders": encoders, "y_encoder": y_enc}, f)
    with open(MODELS / "temperature.pkl", "wb") as f:
        pickle.dump({"T": T_opt}, f)

    # Metrics JSON
    metrics = {
        "data": {"train": len(train), "val": len(val), "test": len(test),
                  "n_features": len(feats)},
        "timings_seconds": timings,
        "val_macro_f1_per_model": val_f1,
        "temperature": T_opt,
        "test": {
            "macro_f1": round(test_f1_macro, 4),
            "weighted_f1": round(test_f1_weighted, 4),
            "accuracy": round(test_acc, 4),
            "log_loss": round(test_ll, 4),
            "ece": round(test_ece, 4),
            "confusion_matrix": confusion_matrix(y_te, pred_te).tolist(),
            "class_names": class_names,
        },
        "fatigue_4bucket": {
            "val_macro_f1": round(float(fat_f1_va), 4),
            "test_macro_f1": round(float(fat_f1_te), 4),
        },
    }
    metrics_name = "final_metrics.json" if args.final else "clean_metrics.json"
    (MODELS / metrics_name).write_text(json.dumps(metrics, indent=2))

    print(f"\nartifacts → {out_dir}/")
    for p in sorted(out_dir.glob("*")):
        print(f"  {p.name:<28} {p.stat().st_size // 1024:>6} KB")
    print(f"\nmetrics  → outputs/models/{metrics_name}")
    print(f"shims    → outputs/models/{{xgb_status.json, tabular_meta.pkl, temperature.pkl}}")


if __name__ == "__main__":
    main()
