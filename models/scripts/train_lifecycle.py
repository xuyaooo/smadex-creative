"""
Lifecycle curve model.

Goal: given the launch-time features of a creative (the same features the
status classifier uses), predict its **14-day CTR curve**, the impressions
and ROAS curves, plus a single fatigue-archetype label.

Outputs:
  outputs/models/lifecycle_xgb.json     XGBoost regressor (14 outputs)
  outputs/models/lifecycle_meta.pkl     feature names, scaler, target stats
  ../front/public/data/lifecycle_curves.json
                                        per-archetype curve table the front
                                        end uses to pick the closest curve
                                        for any predicted creative

Run:
    cd models && PYTHONPATH=$PWD python3 scripts/train_lifecycle.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]      # models/
REPO = ROOT.parent                                # repo root
DATA = REPO / "data"
sys.path.insert(0, str(ROOT))

from src.data.feature_engineering import TabularFeatureEngineer  # noqa: E402

CURVE_LEN = 14            # days
TARGETS = ["ctr", "imps", "roas"]


# ---------------------------------------------------------------------------
def load_daily() -> pd.DataFrame:
    """Aggregate per-day per-creative metrics across countries / OS."""
    df = pd.read_csv(DATA / "creative_daily_country_os_stats.csv")
    g = df.groupby(["creative_id", "days_since_launch"]).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        spend_usd=("spend_usd", "sum"),
        revenue_usd=("revenue_usd", "sum"),
    ).reset_index()
    g["ctr"]  = (g["clicks"]      / g["impressions"].replace(0, np.nan)).fillna(0)
    g["roas"] = (g["revenue_usd"] / g["spend_usd"].replace(0, np.nan)).fillna(0)
    g["imps"] = g["impressions"]
    return g


def build_curves(daily: pd.DataFrame) -> dict[int, dict[str, np.ndarray]]:
    """Per-creative {ctr, imps, roas} arrays of length CURVE_LEN, 0-padded."""
    out: dict[int, dict[str, np.ndarray]] = {}
    for cid, sub in daily.groupby("creative_id"):
        sub = sub.sort_values("days_since_launch")
        s = sub.set_index("days_since_launch").reindex(range(CURVE_LEN)).fillna(0)
        out[int(cid)] = {t: s[t].to_numpy(dtype=np.float32) for t in TARGETS}
    return out


def load_master() -> pd.DataFrame:
    """Use creative_summary as the launch feature source."""
    return pd.read_csv(DATA / "creative_summary.csv")


def normalize_curve(curve: np.ndarray) -> np.ndarray:
    """Scale so the peak day = 1.0 — keeps shape, drops absolute volume.
    For impressions/roas we keep absolute via `_amp` separately."""
    peak = float(curve.max()) if curve.size else 0.0
    return curve / peak if peak > 1e-9 else curve


def archetype_label(ctr: np.ndarray) -> str:
    """Bucket each creative into one of four lifecycle archetypes
    based on where (if anywhere) its CTR collapses."""
    if ctr.max() <= 1e-6:
        return "underperformer"
    n = normalize_curve(ctr)
    # Find the day where CTR has dropped below 70% of peak
    below = np.where(n < 0.7)[0]
    if below.size == 0:                 return "stable"
    first_drop = int(below[0])
    if first_drop <= 4:                 return "early_fatigue"
    if first_drop <= 9:                 return "standard_fatigue"
    return "late_fatigue"


# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading data…")
    daily  = load_daily()
    master = load_master()
    curves = build_curves(daily)
    print(f"  curves built for {len(curves)} creatives")

    # Restrict to creatives that have a curve
    master = master[master["creative_id"].isin(curves.keys())].copy()

    # Build features (re-use the existing TabularFeatureEngineer)
    feng = TabularFeatureEngineer()
    X, feat_names = feng.fit_transform(master)
    cids = master["creative_id"].astype(int).tolist()

    # Build targets: one matrix per metric, shape (n, CURVE_LEN)
    targets: dict[str, np.ndarray] = {}
    for t in TARGETS:
        Y = np.stack([curves[c][t] for c in cids], axis=0)
        targets[t] = Y

    # Per-creative archetype label (off CTR curve)
    archetypes = np.array([archetype_label(curves[c]["ctr"]) for c in cids])
    print("Archetype distribution:")
    for a, n in zip(*np.unique(archetypes, return_counts=True)):
        print(f"  {a:18s} n={n}")

    # Train one XGBoost regressor per metric using sklearn multi-output API
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(cids))
    cut = int(0.85 * len(cids))
    tr, te = idx[:cut], idx[cut:]

    metrics = {}
    fitted: dict[str, list] = {}
    for t in TARGETS:
        # Predict the NORMALIZED shape (peak=1) — preserves curve geometry
        # without requiring the model to predict absolute volume.
        Y = np.stack([normalize_curve(targets[t][i]) for i in range(len(cids))])
        models: list[xgb.XGBRegressor] = []
        preds = np.zeros_like(Y[te])
        for d in range(CURVE_LEN):
            mdl = xgb.XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.06,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                tree_method="hist", random_state=42,
            )
            mdl.fit(Xs[tr], Y[tr, d])
            preds[:, d] = mdl.predict(Xs[te])
            models.append(mdl)
        mae = mean_absolute_error(Y[te], preds)
        r2  = r2_score(Y[te], preds, multioutput="uniform_average")
        print(f"  metric={t:5s}  mae={mae:.3f}  r²={r2:.3f}")
        metrics[t] = {"mae": float(mae), "r2": float(r2)}
        fitted[t] = models

    # Save the XGB models packed into one JSON each + a meta pickle
    art_dir = ROOT / "outputs/models"
    art_dir.mkdir(parents=True, exist_ok=True)
    for t, models in fitted.items():
        booster_dir = art_dir / f"lifecycle_{t}"
        booster_dir.mkdir(exist_ok=True)
        for d, mdl in enumerate(models):
            mdl.save_model(str(booster_dir / f"day{d:02d}.json"))
    with open(art_dir / "lifecycle_meta.pkl", "wb") as f:
        pickle.dump({
            "feat_names": feat_names,
            "scaler": sc,
            "metrics": metrics,
            "curve_len": CURVE_LEN,
            "targets": TARGETS,
        }, f)

    # ----------------------------------------------------------------------
    # Build the front-end lookup table.  For each (vertical × predicted
    # status × archetype) bucket we average the real CTR / imps / ROAS
    # curves and ship that to the SPA — at runtime the page picks the
    # bucket that matches the user's nearest-neighbor prediction and
    # renders the curve.  Falls back to the vertical-level average.
    # ----------------------------------------------------------------------
    # Re-load the precomputed predictions so we know each creative's
    # predicted_status + nearest-neighbor archetype.
    pred_path = REPO / "front/public/data/predictions.json"
    if pred_path.exists():
        preds_corpus = json.loads(pred_path.read_text())
        pred_by_cid = {int(p["creative_id"]): p for p in preds_corpus}
    else:
        pred_by_cid = {}

    bucket: dict[tuple, list[int]] = {}
    for c in cids:
        archetype = archetype_label(curves[c]["ctr"])
        p = pred_by_cid.get(c)
        if p is None:
            continue
        key = (p["vertical"], p["pred_status"], archetype)
        bucket.setdefault(key, []).append(c)

    table: list[dict] = []
    for (v, s, a), members in bucket.items():
        if len(members) < 2:
            continue
        ctr  = np.mean([curves[c]["ctr"]  for c in members], axis=0)
        imps = np.mean([curves[c]["imps"] for c in members], axis=0)
        roas = np.mean([curves[c]["roas"] for c in members], axis=0)
        # Round for JSON readability + size
        table.append({
            "vertical": v,
            "pred_status": s,
            "archetype": a,
            "n": len(members),
            "ctr":  [round(float(x), 5) for x in ctr],
            "imps": [round(float(x), 0) for x in imps],
            "roas": [round(float(x), 4) for x in roas],
        })

    # Vertical-only fallback
    by_vert: dict[str, list[int]] = {}
    for c in cids:
        p = pred_by_cid.get(c)
        if p:
            by_vert.setdefault(p["vertical"], []).append(c)
    fallback = []
    for v, members in by_vert.items():
        ctr  = np.mean([curves[c]["ctr"]  for c in members], axis=0)
        imps = np.mean([curves[c]["imps"] for c in members], axis=0)
        roas = np.mean([curves[c]["roas"] for c in members], axis=0)
        fallback.append({
            "vertical": v, "n": len(members),
            "ctr":  [round(float(x), 5) for x in ctr],
            "imps": [round(float(x), 0) for x in imps],
            "roas": [round(float(x), 4) for x in roas],
        })

    out_path = REPO / "front/public/data/lifecycle_curves.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "curve_len": CURVE_LEN,
        "metrics": metrics,
        "buckets": table,
        "vertical_fallback": fallback,
    }, indent=0))
    print(f"\nWrote {out_path.relative_to(REPO)} — {len(table)} buckets, {len(fallback)} vertical fallbacks")


if __name__ == "__main__":
    main()
