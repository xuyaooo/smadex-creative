# 02 · Data pipeline

> [← 01 · Pipeline overview](01_pipeline_overview.md) · [↑ Index](../README.md) · [03 · Tabular models →](03_tabular_models.md)

How raw CSVs become leakage-free, balanced, group-stratified splits.

**Implementation**:
- [`scripts/build_clean_dataset.py`](../scripts/build_clean_dataset.py) — the whole phase, one CLI
- [`src/data/loader.py`](../src/data/loader.py), [`feature_engineering.py`](../src/data/feature_engineering.py), [`early_features.py`](../src/data/early_features.py) — feature builders
- [`notebooks/01_data_audit.ipynb`](../notebooks/01_data_audit.ipynb) — the bug-hunt that decided what to drop
- [`notebooks/03_dataset_balancing.ipynb`](../notebooks/03_dataset_balancing.ipynb) — narrative walkthrough of splitting + balancing

## Inputs

```
data/
├── creatives.csv               1,080 rows · creative metadata
├── campaigns.csv               180 rows · campaign config
├── advertisers.csv             36 rows · advertiser config
├── creative_summary.csv        per-creative aggregates + creative_status target
├── campaign_summary.csv        per-campaign aggregates
└── creative_daily_country_os_stats.csv   192k rows · the daily fact table
```

## Step 1 — drop bad rows

From [`notebooks/01_data_audit.ipynb`](../notebooks/01_data_audit.ipynb):

```python
BAD_RENDERS = [500204, 500594, 500959]   # text-overflow render bugs
BYTE_DUPS   = [500665]                    # byte-identical dup of 500017
```

Net: **1,076 usable creatives**.

## Step 2 — drop leakage columns (13 total)

```python
# Ratio columns derived from full lifetime
"overall_ctr", "overall_ipm", "overall_roas", "ctr_decay_pct",
"overall_cvr", "cvr_decay_pct", "last_7d_ctr", "last_7d_cvr",
"peak_rolling_ctr_5", "first_7d_ctr", "first_7d_cvr",
# Lifetime cumulative counts — encode the eventual outcome
"total_spend_usd", "total_impressions", "total_clicks",
"total_conversions", "total_revenue_usd", "total_days_active",
"peak_day_impressions",
# Last-7d counts — end-of-life leakage
"last_7d_impressions", "last_7d_clicks", "last_7d_conversions",
# Target proxies
"perf_score", "fatigue_day",
```

**Kept**: `first_7d_impressions`, `first_7d_clicks`, `first_7d_conversions`
— the challenge explicitly expects launch-time predictors, and these are
computable from the first 7 days of fact-table rows alone.

## Step 3 — drop low-signal columns

- `headline`, `subhead`, `cta_text` — only 30 distinct values across 1,080 creatives (vocabulary collapse).
- `text_density`, `readability_score`, `brand_visibility_score`, `clutter_score`, `novelty_score`, `faces_count`, `product_count`, `motion_score` — silent extraction failures (constant-zero on 99% of rows).
- `advertiser_name`, `app_name`, `asset_file` — redundant string IDs.

## Step 4 — merge early-life behavioral features

Computed from `creative_daily_country_os_stats.csv`, restricted to
`days_since_launch ≤ 7`:

```python
early_imp     = sum of impressions in first 7 days
early_clicks  = sum of clicks
early_conv    = sum of conversions
early_spend   = sum of spend_usd
early_revenue = sum of revenue_usd
early_ctr     = early_clicks / early_imp
early_cvr     = early_conv  / early_clicks
```

These are the strongest features in the model (see [03](03_tabular_models.md)).

## Step 5 — KMeans genome cluster (k=24)

Standardised features for clustering:

- 8 categorical: `vertical`, `format`, `theme`, `hook_type`, `dominant_color`, `emotional_tone`, `target_age_segment`, `target_os`
- 4 binary: `has_price`, `has_discount_badge`, `has_gameplay`, `has_ugc_style`
- 4 numeric: `daily_budget_usd`, `duration_sec`, `campaign_duration`, `copy_length_chars`
- 5 early-life features (Step 4)

The cluster id is added as a `cluster` column on every row and used as a
stratification key for the downstream split.

## Step 6 — class-balanced sample weights

```python
sw = compute_sample_weight("balanced", y)        # inverse class frequency
sw[y == "top_performer"] *= 1.7                  # extra boost on rarest
```

Final per-class mean weights:
- `top_performer` 10.16
- `underperformer` 2.83
- `fatigued` 1.36
- `stable` 0.36

## Step 7 — group-aware stratified split

`StratifiedGroupKFold(5)` outer + `StratifiedGroupKFold(6)` inner, grouped
by `campaign_id`, stratified on `vertical | creative_status`. Falls back to
status-only stratification for cells with <2 campaigns.

Result: **train 717 · val 143 · test 216**, with **zero campaign overlap**
across folds.

## Step 8 — outputs

```
outputs/splits/
├── train.parquet                      75 KB · 717 × 31
├── val.parquet                        31 KB · 143 × 31
├── test.parquet                       36 KB · 216 × 31
├── train_balanced_cluster.parquet     83 KB · alt training subset
└── manifest.json                      seed, split sizes, dropped IDs
```

Each parquet has 31 columns: 6 IDs/metadata + 1 target (`creative_status`)
+ 7 categorical + 18 numeric (incl. early-life) + 2 strat columns
(`cluster`, `sample_weight`).

## Design decisions

- **Drop leakage, keep early-life signal.** Lifetime aggregates
  (`overall_*`, `total_*`) encode the eventual outcome — useless at
  launch time. The first-7-day signal is what ad ops actually has on
  day 8, so we keep it.

- **Drop the silently-failing visual columns.** Several dims were
  constant-zero on most rows; the upstream extractor failed quietly.
  Keeping them would just add noise.

- **KMeans cluster as a stratification key.** The `k=24` was tuned
  empirically — small enough to be stable, large enough to capture
  the "vertical × format" structure that drives a lot of variance.

- **Class-balanced weights with an extra boost on `top_performer`.**
  The rarest class needs the largest hand to land at all; the boost
  factor was tuned for max macro-F1, not picked off the shelf.

- **Group-stratified splits, not plain group folds.** With a class
  this rare, plain group folding leaves some folds with zero
  positives. Stratifying inside each group fold keeps every class
  represented everywhere.

## Run it

```bash
PYTHONPATH=models python3 models/scripts/build_clean_dataset.py \
    --k 24 --top-boost 1.7
```

Wall-clock: ~5 s on CPU.

---

[← 01 · Pipeline overview](01_pipeline_overview.md) · [↑ Index](../README.md) · [03 · Tabular models →](03_tabular_models.md)
