# 03 · Tabular models

> [← 02 · Data pipeline](02_data_pipeline.md) · [↑ Index](../README.md) · [04 · Visual intelligence →](04_visual_intelligence.md)

The Model 1 of the chain: the soft-vote ensemble that produces the 4-class
status, the 0–100 Health Score, and the counterfactual lifts. Plus the
fatigue 4-bucket classifier and the lifecycle archetype + curve regressors.

**Implementation**:
- [`scripts/train_clean.py`](../scripts/train_clean.py) — main trainer (val-tuned + `--final` refit)
- [`scripts/train_lifecycle.py`](../scripts/train_lifecycle.py) — 14-day lifecycle curves
- [`src/models/tabular_model.py`](../src/models/tabular_model.py), [`fatigue_detector.py`](../src/models/fatigue_detector.py) — model wrappers
- [`src/fatigue/health_score.py`](../src/fatigue/health_score.py) — the 0–100 blend
- [`notebooks/04_models.ipynb`](../notebooks/04_models.ipynb) — train + benchmark + TreeSHAP + counterfactuals

## Status classifier — soft-vote ensemble of 5

| Model | Library | File (`outputs/models/clean/`) | File (`outputs/models/final/`) |
|---|---|---|---|
| XGBoost 5-seed bag | `xgboost` | `xgb_seed{0..4}.json` (~2.6 MB each) | `xgb_seed{0..4}.json` (~2.7 MB each) |
| LightGBM | `lightgbm` | `lgb.pkl` (3.9 MB) | `lgb.pkl` (3.9 MB) |
| CatBoost | `catboost` | `catboost.cbm` (808 KB) | `catboost.cbm` (823 KB) |
| HistGradientBoosting | `sklearn` | `hgb.pkl` (2.0 MB) | `hgb.pkl` (2.0 MB) |
| Logistic Regression | `sklearn` | `logreg.pkl` (8 KB) | `logreg.pkl` (8 KB) |
| Encoders + label map | — | `meta.pkl` (2 KB) | `meta.pkl` (2 KB) |

**Aggregation**: arithmetic mean of per-model softmax probabilities.

**No temperature scaling** — `T = 1.0`. We tried Guo-style temperature
scaling; the optimal `T ≈ 0.96` was indistinguishable from a no-op so we
removed it. ECE on the held-out test = 0.073, in the "acceptable but
visibly miscalibrated" band.

### XGBoost hyperparameters

```python
n_estimators = 500
max_depth    = 4
learning_rate = 0.04
subsample    = 0.85
colsample_bytree = 0.75
seeds        = [42, 1, 2, 3, 4]
```

### Two trained variants

| Variant | Trained on | When to use | Where |
|---|---|---|---|
| `clean/` | train (717) | val-tuned baseline, model selection | `outputs/models/clean/` |
| `final/` | train ∪ val (860) | production refit, the one shipped | `outputs/models/final/` |

```bash
python3 scripts/train_clean.py            # ~17 s — writes outputs/models/clean/
python3 scripts/train_clean.py --final    # ~20 s — writes outputs/models/final/
```

## Health Score (0–100)

The user-facing score that drives the **Scale / Continue / Pivot / Pause**
recommendation tiers. Implemented in [`src/fatigue/health_score.py`](../src/fatigue/health_score.py).

It blends:
- ensemble class probabilities (top_performer ↑, underperformer ↓)
- fatigue 4-bucket prob (early/standard fatigue ↓)
- early-life CTR z-score within vertical (regularises by cohort)

Calibration on test:
- Health Score → status Spearman = **0.452** (p < 1e-11).
- Per-class means: `top_performer 84/100 · stable 50 · fatigued 50 · underperformer 19`.
- Tier precision: **Scale (≥75) precision 0.78** (7/9 picks are real top performers); Pause (<25) precision 0.54.

## Fatigue 4-bucket classifier

Targets: **Never · Late · Standard · Early**.

- Model: `LightGBM(num_leaves=15, class_weight='balanced')`
- File: `outputs/models/{clean,final}/fatigue_4bucket.pkl` (2.1 MB)
- Test macro-F1 (final): **0.428**; binary fatigued/not-fatigued ≈ **0.85** F1
- The Late bucket is the hardest (F1 ≈ 0)

## Lifecycle predictor

Two heads, both trained by [`scripts/train_lifecycle.py`](../scripts/train_lifecycle.py):

1. **Archetype classifier** — 5 lifecycle shapes (stable, late_fatigue, standard_fatigue, early_fatigue, underperformer). Val macro-F1 = **0.617** vs 0.20 baseline (5×).
2. **14-day curve heads** — per-day predicted `ctr / impressions / roas`. Per-sample r² ≈ 0.07; we ship the **bucket-mean curves** of real top performers in production because they are more honest than the regressor's per-sample predictions.

## Counterfactual lifts

For each prediction we compute single-feature counterfactuals on the
ensemble: hold the row constant, sweep one feature across its plausible
range, and surface the top-3 lifts above a noise floor. These power the
"if you change X to Y, predicted health goes from A → B" line on the
front-end's Predict page.

Implemented inline in `train_clean.py` and reproduced in
[`notebooks/04_models.ipynb`](../notebooks/04_models.ipynb).

## Design decisions

- **Five models, not one.** Each library has its own inductive bias —
  trees vs. linear, level-wise vs. leaf-wise, native categoricals vs.
  encoded ones. Averaging them smooths out each one's blind spots.
  Removing any single member costs a small but real chunk of macro-F1.

- **Five seeds for XGBoost.** It's the highest-variance member across
  seeds, so the bag pays off there. The other libraries are stable
  enough at this size that more seeds wouldn't help.

- **Arithmetic mean of probabilities, not a learned stacker.** A
  stacker would need its own held-out set, which the rare classes
  can't really afford here. The simpler average is the safer call
  and was within noise of the alternatives we tried.

- **No temperature scaling in production.** The optimum scalar landed
  essentially on the no-op. Averaging several probabilistic models
  is itself a calibration regulariser, so the ensemble was already
  close enough.

- **Fatigue as a separate model, not a multi-task head.** Fatigue's
  target only makes sense for non-stable creatives, and a shared
  trunk made the status head a touch worse. Two clean models beat
  one tangled one.

- **Ship bucket-mean lifecycle curves, not the regressor.** The
  per-sample regressor catches the average shape but not the
  individual one. Serving real top-performer curve means by
  vertical × format is the more honest answer.

---

[← 02 · Data pipeline](02_data_pipeline.md) · [↑ Index](../README.md) · [04 · Visual intelligence →](04_visual_intelligence.md)
