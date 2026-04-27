# 06 · Evaluation

> [← 05 · Personalized VLM + Flux edit](05_personalized_vlm_and_image.md) · [↑ Index](../README.md)

How the numbers in `01_pipeline_overview.md` are produced and how to
reproduce them. The held-out test set is the same one carved by
[`scripts/build_clean_dataset.py`](../scripts/build_clean_dataset.py)
(216 creatives, no campaign overlap with train/val).

**Implementation**:
- [`notebooks/05_evals.ipynb`](../notebooks/05_evals.ipynb) — comprehensive eval (per-class CIs, per-vertical, ROC, reliability, action-tier crosstab)
- [`eval.py`](../eval.py) — end-to-end metric + latency report (used by CI)
- [`src/evaluation/`](../src/evaluation/) — metric helpers

## Headline metrics (held-out test, n=216)

| Metric | Val-tuned (n=717) | **Final (n=860)** | Δ |
|---|---|---|---|
| macro-F1 | 0.6345 | **0.6774** | +4.3pp |
| weighted-F1 | 0.7578 | **0.7807** | +2.3pp |
| accuracy | 0.7500 | **0.7731** | +2.3pp |
| log-loss | 0.5800 | **0.5767** | −0.003 |
| ECE | 0.0779 | **0.0731** | −0.005 |

`final/` = ensemble refit on train ∪ val (n=860), evaluated on test.
This is the variant shipped to production.

## Per-class (final)

| Class | n_test | F1 | F1 95% CI | precision | recall | AUC |
|---|---|---|---|---|---|---|
| stable | 154 | 0.84 | [0.79, 0.87] | 0.89 | 0.80 | — |
| fatigued | 35 | 0.63 | [0.49, 0.74] | 0.57 | 0.71 | — |
| underperformer | 16 | 0.63 | [0.39, 0.87] | 0.52 | 0.81 | **0.975** |
| top_performer | 11 | 0.60 | [0.31, 0.89] | 0.67 | 0.55 | **0.941** |

CIs are bootstrap (1000 resamples, stratified by class).

## Per-vertical (test)

| Vertical | n | macro-F1 | accuracy |
|---|---|---|---|
| travel | 36 | 0.725 | 0.722 |
| gaming | 36 | 0.700 | 0.722 |
| fintech | 36 | 0.660 | 0.917 |
| food_delivery | 36 | 0.560 | 0.806 |
| entertainment | 36 | 0.470 | 0.778 |
| ecommerce | 36 | 0.410 | 0.833 |

## Heads beyond status

- **Health Score → status Spearman**: 0.452 (p < 1e-11). Per-class means: top 84/100 · stable 50 · fatigued 50 · under 19.
- **Health Score → action precision**: Scale tier (≥75) = **0.78** (7 / 9 picks are real top performers). Pause tier (<25) = 0.54.
- **Fatigue 4-bucket macro-F1**: 0.428 (Never F1 = 0.90; Late F1 = 0; binary fatigued/not-fatigued ≈ 0.85 F1).
- **Lifecycle archetype predictor**: val macro-F1 = 0.617 vs 1/5 = 0.20 baseline — 3× chance.

## Calibration

- **ECE** = 0.0731 — "acceptable but visibly miscalibrated" band.
- **Temperature scaling** was tried (Guo ICML 2017, single scalar T fit on val log-likelihood). Optimal T ≈ 0.96 ≈ no-op, so it is **not used**. The shipped pipeline returns raw softmax probabilities.
- For trees, Beta calibration (Kull AISTATS 2017) is the more appropriate next step; not yet shipped.

## Latency (30-creative sample, single-process, warm cache)

| Operation | mean | p50 | p95 | max | Threshold |
|---|---:|---:|---:|---:|---|
| `health_score` | 44.7 ms | 43.1 | 53.4 | 58.8 | p95 < 100 ms ✓ |
| `find_similar` | 12.1 ms | 11.9 | 12.4 | 17.2 | p95 < 50 ms ✓ |
| `cluster_info` | 0.1 ms | 0.1 | 0.1 | 0.2 | — |
| `explain` | 66.6 ms | 62.9 | 77.1 | 125.4 | p95 < 150 ms ✓ |
| Cold start | 2.02 s | — | — | — | < 5 s ✓ |

Run them yourself:

```bash
PYTHONPATH=models python3 models/eval.py
```

## Honest caveats

1. **The dataset is synthetic** with a documented `vertical → status` confound (χ² ≈ 335). A predict-by-vertical-prior baseline hits ~0.30 macro-F1; the model gets 0.677, so we contribute **+0.35 macro-F1 over the prior**, mostly from `early_ctr`.
2. **Rare-class point estimates have wide CIs**. With n=11 top_performers and n=16 underperformers in test, single-flip changes move F1 by 0.05–0.08. Always report CIs.
3. **Pause / Pivot recommendations are not yet autonomous-safe** (precision 0.54, industry bar 0.75). Surface as a *Watch* queue, not an auto-action.
4. **Visual side adds ≤ 3pp** macro-F1 on top of the tabular baseline. The synthetic generator's visual content has near-identical wireframes for top vs underperformers, so expect bigger lifts on real data.
5. **Lifecycle curve XGBoost regressor has r² ≈ 0.07** on shape; we ship the **bucket-means** of real curves, not the regressor's per-sample prediction, because the bucket means are more honest.
6. **Cold-start**: the 7-day early-life signal is the strongest feature. For brand-new creatives with zero impressions the model degrades to the visual-rubric + metadata baseline (~0.45 macro-F1).

## What's fair to claim in a workshop demo

- Tabular-only macro-F1 **0.677** on a clean, leakage-free, group-stratified split — competitive with TabPFN-v2 / TabM / CatBoost benchmarks on n < 1k 4-class imbalanced problems.
- Top-performer detection **AUC 0.94**, underperformer **AUC 0.98**.
- Health-Score → Scale recommendation has **78% precision** (7 of 9 picks are actual winners).
- End-to-end Model 1 reproducible in **~4 minutes on CPU** from raw CSVs.

## Design decisions

- **Bootstrap CIs over analytical ones.** With this many tiny per-class
  cells, closed-form intervals lie. Resampling is the right call — a
  bit slower, much more honest.

- **Hold-out test, not k-fold cross-val.** Group-stratified splits at
  this dataset size already eat enough rare-class signal; doing it
  five times wouldn't tell us anything new and would just rotate
  noise.

- **Report calibration, don't fix it artificially.** The shipped model
  has a real but small calibration gap. Forcing a correction that
  isn't earned (the temperature scaler's optimum was a no-op) would
  paper over it without changing behaviour.

- **One headline number, then caveats.** macro-F1 is what the brief
  asks for, so we lead with it. The caveats live alongside, not
  somewhere quieter — wide rare-class CIs and the synthetic-data
  baseline are part of the same story.

---

[← 05 · Personalized VLM + Flux edit](05_personalized_vlm_and_image.md) · [↑ Index](../README.md)
