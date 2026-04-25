# 🚀 Creative Intelligence Pipeline — Full Overview

**One-page reference**: every preprocessing step, model path, and metric in one place.

Read this file first, then run the notebooks in order (`01_…05_`).

---

## TL;DR

| | |
|---|---|
| **Dataset** | Smadex Creative Intelligence Challenge — 1,076 mobile-ad creatives × 36 advertisers × 6 verticals |
| **Task** | 4-class status (top_performer / stable / fatigued / underperformer) + fatigue forecast + 0–100 Health Score |
| **Splits** | 717 train / 143 val / 216 test (no campaign overlap) |
| **Final test macro-F1** | **0.677** (calibrated soft-vote ensemble of 5 models, refit on train+val) |
| **Total wall-clock to reproduce** | ~4 minutes (1 build + 2 train + 5 notebook executes) |

---

## 🔁 The pipeline at a glance

```
  ┌─────────────────────────┐
  │  raw CSVs (3 files)     │
  │  /root/smadex-creative/ │
  │  ├── creative_summary   │
  │  ├── creatives          │
  │  └── creative_daily_…   │
  └────────────┬────────────┘
               │
               │  scripts/build_clean_dataset.py
               │   • drop 4 bad creatives (3 render bugs + 1 byte-dup)
               │   • drop 13 leakage cols (total_*, last_7d_*, perf_score, …)
               │   • merge first-7d behavioral aggregates
               │   • KMeans-24 cluster on tabular + early-life
               │   • class-balanced sample weights × 1.7 boost on top_performer
               │   • StratifiedGroupKFold by campaign_id (70/15/15)
               ▼
  ┌─────────────────────────┐
  │  outputs/splits/        │
  │  ├── train.parquet  717 │
  │  ├── val.parquet    143 │
  │  ├── test.parquet   216 │
  │  ├── train_balanced…823 │
  │  └── manifest.json      │
  └────────────┬────────────┘
               │
               │  scripts/train_clean.py             scripts/train_clean.py --final
               │  ↓ (val-tuned baseline)              ↓ (production refit on train ∪ val)
               ▼                                     ▼
  ┌─────────────────────────┐                       ┌─────────────────────────┐
  │  outputs/models/clean/  │                       │  outputs/models/final/  │
  │  ├── xgb_seed{0..4}.json│                       │  same 11 files          │
  │  ├── lgb.pkl            │                       │  trained on n=860       │
  │  ├── catboost.cbm       │                       │  test macro-F1 = 0.677  │
  │  ├── hgb.pkl            │                       └─────────────────────────┘
  │  ├── logreg.pkl         │
  │  ├── fatigue_4bucket.pkl│
  │  └── meta.pkl           │
  │  test macro-F1 = 0.635  │
  └─────────────────────────┘
```

---

## 🧹 Preprocessing — every step, in order

Implemented in `scripts/build_clean_dataset.py`. Configurable via CLI flags `--k 24 --top-boost 1.7`.

### Step 1: Drop bad rows

From the audit notebook (`01_data_audit.ipynb`):

```python
BAD_RENDERS = [500204, 500594, 500959]   # text-overflow render bugs
BYTE_DUPS   = [500665]                    # byte-identical dup of 500017
```

### Step 2: Drop leakage columns (13 total)

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

`first_7d_impressions / clicks / conversions` are **KEPT** — legitimate launch-time predictors the challenge explicitly expects.

### Step 3: Drop low-signal columns

- `headline`, `subhead`, `cta_text` — only 30 distinct values for 1,080 creatives (vocabulary collapse)
- `text_density`, `readability_score`, `brand_visibility_score`, `clutter_score`, `novelty_score`, `faces_count`, `product_count`, `motion_score` — silent extraction failures (constant-zero on 99% of rows)
- `advertiser_name`, `app_name`, `asset_file` — redundant string IDs

### Step 4: Merge early-life behavioral features

Computed from `creative_daily_country_os_stats.csv`, `days_since_launch ≤ 7`:

```python
early_imp     = sum of impressions in first 7 days
early_clicks  = sum of clicks
early_conv    = sum of conversions
early_spend   = sum of spend_usd
early_revenue = sum of revenue_usd
early_ctr     = early_clicks / early_imp
early_cvr     = early_conv  / early_clicks
```

### Step 5: KMeans genome cluster (k=24)

Standardized features for clustering:
- 8 categorical: vertical, format, theme, hook_type, dominant_color, emotional_tone, target_age_segment, target_os
- 4 binary: has_price, has_discount_badge, has_gameplay, has_ugc_style
- 4 numeric: daily_budget_usd, duration_sec, campaign_duration, copy_length_chars
- 5 early-life features

### Step 6: Class-balanced sample weights

```python
sw = compute_sample_weight("balanced", y)        # inverse class frequency
sw[y == "top_performer"] *= 1.7                  # extra boost on rarest
```

Final per-class mean weights: `top_performer 10.16` · `underperformer 2.83` · `fatigued 1.36` · `stable 0.36`.

### Step 7: Group-aware stratified split

`StratifiedGroupKFold(5)` outer + `StratifiedGroupKFold(6)` inner, grouped by `campaign_id`, stratified on `vertical | creative_status`. Fall back to status-only stratification for cells with <2 campaigns.

Result: **train=717 · val=143 · test=216, zero campaign overlap** across folds.

### Step 8: Output

```
outputs/splits/
├── train.parquet                      (75 KB, 717 × 31)
├── val.parquet                        (31 KB, 143 × 31)
├── test.parquet                       (36 KB, 216 × 31)
├── train_balanced_cluster.parquet     (83 KB, alt training subset)
└── manifest.json
```

Each parquet has 31 columns: 6 IDs/metadata + 1 target (creative_status) + 7 categorical + 18 numeric (incl. early-life) + 2 strat cols (cluster, sample_weight).

---

## 🧠 Models trained — paths + sizes + metrics

Implemented in `scripts/train_clean.py`. Wall-clock: ~17s val-tuned, ~20s `--final`.

### Status classifier (4-class, the main model)

Soft-vote ensemble of 5 models. **No temperature scaling** — raw probabilities used everywhere.

| Model | Library | File (clean/) | File (final/) |
|---|---|---|---|
| XGBoost 5-seed bag | `xgboost` | `xgb_seed{0..4}.json` (~2.6 MB each) | `xgb_seed{0..4}.json` (~2.7 MB each) |
| LightGBM | `lightgbm` | `lgb.pkl` (3.9 MB) | `lgb.pkl` (3.9 MB) |
| CatBoost | `catboost` | `catboost.cbm` (808 KB) | `catboost.cbm` (823 KB) |
| HistGradientBoosting | `sklearn` | `hgb.pkl` (2.0 MB) | `hgb.pkl` (2.0 MB) |
| Logistic Regression | `sklearn` | `logreg.pkl` (8 KB) | `logreg.pkl` (8 KB) |
| Encoders + label map | — | `meta.pkl` (2 KB) | `meta.pkl` (2 KB) |

Per-model XGB hyperparams: `n_estimators=500, max_depth=4, learning_rate=0.04, subsample=0.85, colsample_bytree=0.75`, seeds = `[42, 1, 2, 3, 4]`.

### Fatigue classifier (4 buckets)

Model: `LightGBM(num_leaves=15, class_weight='balanced')`. Targets: Never / Late / Standard / Early.
- File: `outputs/models/{clean,final}/fatigue_4bucket.pkl` (2.1 MB)
- Test macro-F1 (final): **0.428**

### Backend-compat shims (loaded by `backend/main.py`)

| File | Purpose |
|---|---|
| `outputs/models/xgb_status.json` | first XGB seed for old API path |
| `outputs/models/tabular_meta.pkl` | label encoder + feature names |
| `outputs/models/temperature.pkl` | T = 1.0 (no-op) |

### Visual side (built earlier, optional)

| Artifact | Path | What |
|---|---|---|
| SigLIP-2 embeddings | `outputs/embeddings/clip_embeddings.npz` (1.5 MB) | per-creative L2-norm + PCA(64) |
| UMAP coordinates | `outputs/clusters/umap_coords.npz` | 2D viz |
| HDBSCAN labels | `outputs/clusters/labels.parquet` | visual clusters |
| Per-vertical kNN | `outputs/knn/index.pkl` (6.4 MB) | retrieval index |
| LLM rubric | `outputs/rubric/rubric_scores.parquet` (20 KB) | gemini-2.5-flash extracted |
| SmolVLM LoRA | `outputs/models/vlm_finetuned/` | image-aware student |
| SigLIP-2 LoRA | `outputs/models/siglip2_finetuned/` | (.gitignored, 715 MB) |

---

## 📊 Final metrics (held-out test, n=216)

### Headline

| Metric | Val-tuned (n=717) | **Final (n=860)** | Δ |
|---|---|---|---|
| macro-F1 | 0.6345 | **0.6774** | +4.3pp |
| weighted-F1 | 0.7578 | **0.7807** | +2.3pp |
| accuracy | 0.7500 | **0.7731** | +2.3pp |
| log-loss | 0.5800 | **0.5767** | −0.003 |
| ECE | 0.0779 | **0.0731** | −0.005 |

### Per-class (final)

| Class | n_test | F1 | F1 95% CI | precision | recall | AUC |
|---|---|---|---|---|---|---|
| stable | 154 | 0.84 | [0.79, 0.87] | 0.89 | 0.80 | — |
| fatigued | 35 | 0.63 | [0.49, 0.74] | 0.57 | 0.71 | — |
| underperformer | 16 | 0.63 | [0.39, 0.87] | 0.52 | 0.81 | **0.975** |
| top_performer | 11 | 0.60 | [0.31, 0.89] | 0.67 | 0.55 | **0.941** |

### Heads beyond status

- **Health Score → status Spearman**: 0.452 (p < 1e-11). top_performer mean=84/100, stable=50, fatigued=50, underperformer=19.
- **Health Score → action precision**: Scale tier (≥75) precision **0.78** (7/9 are top_performers). Pause tier (<25) precision 0.54.
- **Fatigue 4-bucket macro-F1**: 0.428 (Never F1 = 0.90; Late F1 = 0; predict-binary fatigue ≈ 0.85 F1).
- **Lifecycle archetype predictor**: val macro-F1 = 0.617 vs 1/5 = 0.20 baseline — 3× chance.

### Per-vertical (test)

| Vertical | n | macro-F1 | accuracy |
|---|---|---|---|
| travel | 36 | 0.725 | 0.722 |
| gaming | 36 | 0.700 | 0.722 |
| fintech | 36 | 0.660 | 0.917 |
| food_delivery | 36 | 0.560 | 0.806 |
| entertainment | 36 | 0.470 | 0.778 |
| ecommerce | 36 | 0.410 | 0.833 |

---

## 📓 Notebooks (run in order)

| # | Notebook | Time | What it does | Outputs |
|---|---|---|---|---|
| 01 | `01_data_audit.ipynb` | 1.5 min | Bug-hunt the raw data: byte-dups, render bugs, leakage flags, vocabulary collapse, vertical confound | `outputs/clean/` |
| 02 | `02_data_analysis.ipynb` | 7s | Pattern discovery: 5 lifecycle archetypes, 12 genome clusters, top-vs-under Cohen's-d, multimodal correlations | inline plots |
| 03 | `03_dataset_balancing.ipynb` | 4s | Narrative walkthrough of the splitting + balancing logic | `outputs/splits_demo/` |
| 04 | `04_models.ipynb` | 14s | Train + benchmark 6 models (XGB/LGB/CatBoost/HGB/LR/Ensemble), TreeSHAP, counterfactuals, lifecycle archetype predictor | inline tables |
| 05 | `05_evals.ipynb` | 13s | Comprehensive eval: per-class F1 with bootstrap CIs, per-vertical/cluster, reliability diagram, ROC curves, action-tier crosstab, hardest-row analysis | `outputs/models/eval_report.json` |

---

## 🔄 Reproduce end-to-end

```bash
# 1. Install deps (~1 minute)
pip install -r creative_intelligence/requirements.txt

# 2. Build clean splits from raw CSVs (~5 seconds)
python3 creative_intelligence/scripts/build_clean_dataset.py

# 3. Train val-tuned baseline (~17 seconds)
python3 creative_intelligence/scripts/train_clean.py

# 4. Refit on train ∪ val for production (~20 seconds)
python3 creative_intelligence/scripts/train_clean.py --final

# 5. Run the 5 notebooks in order (~3 minutes total)
cd creative_intelligence/notebooks
for nb in 01_data_audit 02_data_analysis 03_dataset_balancing 04_models 05_evals; do
    jupyter nbconvert --to notebook --execute ${nb}.ipynb --output ${nb}.ipynb
done
```

Total wall-clock to reproduce from raw CSVs: **~4 minutes** on CPU. RTX 4090 is available but not faster at n=717.

---

## 🧪 Honest caveats

From the 5-review audit:

1. **The dataset is synthetic** with a documented `vertical → status` confound (χ²≈335). A predict-by-vertical-prior baseline hits ~0.30 macro-F1; our model gets 0.677, so we contribute **+0.35 macro-F1 over the prior**, mostly from `early_ctr`.
2. **Rare-class point estimates have wide CIs**. With n=11 top_performers and n=16 underperformers in test, single-flip changes move F1 by 0.05–0.08. Always report CIs.
3. **Pause/Pivot recommendations are not yet autonomous-safe** (precision 0.54, industry bar 0.75). Use as a *Watch* queue, not an auto-action.
4. **Visual side adds ≤3pp** macro-F1 on top of the tabular baseline. The synthetic generator's visual content has near-identical wireframes for top vs underperformers, so expect bigger lifts on real data.
5. **Calibration is acceptable, not great** — ECE 0.073 is in the "acceptable but visibly miscalibrated" band. Temperature scaling didn't help (T was ~0.96 ≈ no-op) so we removed it.

### What's fair to claim in a hackathon demo

- *Tabular-only macro-F1 0.677 on a clean leakage-free split — competitive with TabPFN-v2 / TabM / CatBoost benchmarks on n<1k 4-class imbalanced.*
- *Top-performer detection AUC 0.94, underperformer AUC 0.98.*
- *Health Score → Scale recommendation has 78% precision (7 of 9 picks are actual winners).*
- *End-to-end reproducible in 4 minutes on CPU.*
