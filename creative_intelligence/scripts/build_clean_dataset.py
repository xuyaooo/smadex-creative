"""
Production cleaning + balancing pipeline → train/val/test splits.

Produces `outputs/splits/{train,val,test}.parquet` (+ a balanced-cluster
alternative training set) ready for `scripts/train_all.py` to consume.

Steps:
  1. Drop the 4 known-bad creatives (3 render bugs + 1 byte-identical dup).
  2. Drop leakage / vocab-collapsed / low-MI columns.
  3. KMeans-cluster on early-life + tabular features.
  4. Compute class-balanced sample weights (with extra boost on top_performer).
  5. StratifiedGroupKFold split: 70 train / 15 val / 15 test, stratified on
     (vertical | creative_status), grouped by campaign_id.

Usage:
    python3 scripts/build_clean_dataset.py
    python3 scripts/build_clean_dataset.py --k 32 --top-boost 2.0
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight


# Drop list pulled from the audit notebook (data_audit.ipynb).
BAD_RENDERS = [500204, 500594, 500959]   # text-overflow render bugs
BYTE_DUPS = [500665]                     # duplicate of 500017

DROP_LEAKAGE = [
    "overall_ctr", "overall_ipm", "overall_roas", "ctr_decay_pct",
    "overall_cvr", "cvr_decay_pct", "last_7d_ctr", "last_7d_cvr",
    "peak_rolling_ctr_5", "first_7d_ctr", "first_7d_cvr",
]
DROP_VOCAB = ["headline", "subhead", "cta_text"]
DROP_LOWMI = [
    "text_density", "readability_score", "brand_visibility_score",
    "clutter_score", "novelty_score", "faces_count", "product_count",
    "motion_score",
]
DROP_REDUNDANT = ["advertiser_name", "app_name", "asset_file"]


def load_raw(repo: Path):
    data = repo.parent
    return {
        "summary":   pd.read_csv(data / "creative_summary.csv"),
        "creatives": pd.read_csv(data / "creatives.csv"),
        "daily":     pd.read_csv(data / "creative_daily_country_os_stats.csv",
                                  parse_dates=["date"]),
    }


def drop_bad_rows(summary: pd.DataFrame, exclude: set[int]) -> pd.DataFrame:
    return summary[~summary.creative_id.isin(exclude)].reset_index(drop=True)


def drop_bad_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col_set in (DROP_LEAKAGE, DROP_VOCAB, DROP_LOWMI, DROP_REDUNDANT):
        out = out.drop(columns=[c for c in col_set if c in out.columns])
    return out


def early_life_features(daily: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    sub = daily[daily.days_since_launch <= window]
    out = (
        sub.groupby("creative_id")
        .agg(
            early_imp=("impressions", "sum"),
            early_clicks=("clicks", "sum"),
            early_conv=("conversions", "sum"),
            early_spend=("spend_usd", "sum"),
            early_revenue=("revenue_usd", "sum"),
        )
        .reset_index()
    )
    out["early_ctr"] = out.early_clicks / out.early_imp.replace(0, np.nan)
    out["early_cvr"] = out.early_conv / out.early_clicks.replace(0, np.nan)
    return out.fillna(0)


def cluster_genome(summary: pd.DataFrame, daily: pd.DataFrame,
                    k: int, seed: int) -> tuple[np.ndarray, list[str]]:
    cat_cols = ["vertical", "format", "theme", "hook_type",
                "dominant_color", "emotional_tone", "target_age_segment", "target_os"]
    bin_cols = ["has_price", "has_discount_badge", "has_gameplay", "has_ugc_style"]
    num_cols = ["daily_budget_usd", "duration_sec", "campaign_duration", "copy_length_chars"]

    feat = summary.copy()
    if "campaign_duration" not in feat.columns:
        if "start_date" in feat.columns and "end_date" in feat.columns:
            feat["campaign_duration"] = (
                pd.to_datetime(feat.end_date) - pd.to_datetime(feat.start_date)
            ).dt.days
        else:
            feat["campaign_duration"] = 0

    for c in cat_cols:
        feat[c] = feat.get(c, "NA")
        feat[c] = LabelEncoder().fit_transform(feat[c].fillna("NA").astype(str))
    for c in bin_cols + num_cols:
        if c not in feat.columns:
            feat[c] = 0

    early = early_life_features(daily)
    feat = feat.merge(early, on="creative_id", how="left").fillna(0)

    feature_columns = cat_cols + bin_cols + num_cols + [
        "early_imp", "early_clicks", "early_ctr", "early_cvr", "early_revenue",
    ]
    X = feat[feature_columns].astype(np.float32).values
    Xs = StandardScaler().fit_transform(X)
    labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Xs)
    return labels, feature_columns


def per_cluster_balance(summary: pd.DataFrame, max_per_class: int,
                         seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for _, group in summary.groupby("cluster"):
        for _, sub in group.groupby("creative_status"):
            target = min(len(sub), max_per_class)
            if len(sub) > target:
                parts.append(sub.sample(n=target, random_state=int(rng.integers(2**31))))
            else:
                parts.append(sub)
    return pd.concat(parts).reset_index(drop=True)


def stratified_group_split(summary: pd.DataFrame, seed: int = 42):
    """Returns (train_idx, val_idx, test_idx) indexed into `summary`."""
    summary = summary.copy()
    summary["strat_key"] = summary.vertical.astype(str) + "|" + summary.creative_status.astype(str)

    # Some (vertical, status) cells have <2 campaigns — fall back to status-only there
    cells = (summary[["campaign_id", "strat_key"]]
             .drop_duplicates()["strat_key"].value_counts())
    rare = cells[cells < 2].index.tolist()
    if rare:
        mask = summary.strat_key.isin(rare)
        summary.loc[mask, "strat_key"] = summary.loc[mask, "creative_status"]

    y = summary.strat_key.values
    g = summary.campaign_id.values

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    trainval_idx, test_idx = next(outer.split(summary, y, g))

    tv = summary.iloc[trainval_idx].reset_index(drop=False)
    tv_y = tv.strat_key.values
    tv_g = tv.campaign_id.values
    inner = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    train_local, val_local = next(inner.split(tv, tv_y, tv_g))

    train_idx = tv.iloc[train_local]["index"].values
    val_idx = tv.iloc[val_local]["index"].values
    return train_idx, val_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=24, help="number of KMeans clusters")
    parser.add_argument("--top-boost", type=float, default=1.7,
                        help="extra multiplicative weight on top_performer rows")
    parser.add_argument("--max-per-class", type=int, default=25,
                        help="cap per (cluster, class) cell in the balanced subset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    repo = Path(__file__).parent.parent
    out_dir = repo / "outputs/splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading raw CSVs...")
    raw = load_raw(repo)
    summary = raw["summary"]
    daily = raw["daily"]

    print(f"  rows in: {len(summary)}")

    # 1. Drop bad rows
    exclude = set(BAD_RENDERS) | set(BYTE_DUPS)
    summary = drop_bad_rows(summary, exclude)
    print(f"  after dropping {len(exclude)} bad creatives: {len(summary)}")

    # 2. Drop bad columns (informational — keep on the parquet so downstream
    #    modules can still grab metadata they expect, but compute the model
    #    feature frame from a cleaned subset)
    summary_clean = drop_bad_columns(summary)
    print(f"  columns: {summary.shape[1]} → {summary_clean.shape[1]}")

    # 3. Cluster
    labels, feat_cols = cluster_genome(summary_clean, daily, k=args.k, seed=args.seed)
    summary_clean["cluster"] = labels
    print(f"  KMeans({args.k}): cluster sizes "
          f"min={pd.Series(labels).value_counts().min()}, "
          f"max={pd.Series(labels).value_counts().max()}")

    # 4. Class-balanced sample weights
    sw = compute_sample_weight("balanced", summary_clean.creative_status)
    sw[summary_clean.creative_status == "top_performer"] *= args.top_boost
    summary_clean["sample_weight"] = sw

    # 5. Per-cluster balanced subsample (alternative training set)
    balanced = per_cluster_balance(summary_clean, args.max_per_class, args.seed)

    # 6. Splits
    train_idx, val_idx, test_idx = stratified_group_split(summary_clean, seed=args.seed)
    train = summary_clean.iloc[train_idx].reset_index(drop=True)
    val = summary_clean.iloc[val_idx].reset_index(drop=True)
    test = summary_clean.iloc[test_idx].reset_index(drop=True)

    print(f"  train: {len(train)}   val: {len(val)}   test: {len(test)}")
    overlap = {
        "train_val":  len(set(train.campaign_id) & set(val.campaign_id)),
        "train_test": len(set(train.campaign_id) & set(test.campaign_id)),
        "val_test":   len(set(val.campaign_id) & set(test.campaign_id)),
    }
    print(f"  campaign overlap: {overlap}")
    assert sum(overlap.values()) == 0, "campaigns leaked across folds!"

    # Write outputs
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    balanced.to_parquet(out_dir / "train_balanced_cluster.parquet", index=False)

    manifest = {
        "rows_excluded": sorted(exclude),
        "exclusion_reasons": {
            "render_bugs": BAD_RENDERS,
            "byte_identical_dups": BYTE_DUPS,
        },
        "columns_dropped": {
            "leakage": DROP_LEAKAGE,
            "vocab_collapse": DROP_VOCAB,
            "low_mi_visual": DROP_LOWMI,
            "redundant_strings": DROP_REDUNDANT,
        },
        "clustering": {"method": "KMeans", "n_clusters": args.k,
                        "feature_columns": feat_cols},
        "sample_weight_strategy": {
            "base": "balanced", "top_performer_extra_boost": args.top_boost,
        },
        "splits": {"train": len(train), "val": len(val), "test": len(test)},
        "balanced_subsample_size": len(balanced),
        "campaign_overlap": overlap,
        "files": {
            "train.parquet":              "training set with sample_weight column",
            "val.parquet":                "validation set, natural class distribution",
            "test.parquet":               "held-out test, touch ONCE at the end",
            "train_balanced_cluster.parquet": "alt training set, per-cluster subsampled",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\nfiles written:")
    for p in sorted(out_dir.glob("*")):
        print(f"  {p.name:<40} {p.stat().st_size // 1024:>6} KB")


if __name__ == "__main__":
    main()
