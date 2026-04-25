# QA Report — Creative Intelligence Pipeline

**Date:** 2026-04-25
**Reviewer:** QA agent (post-5-parallel-improvement-agent run)
**Repo:** `/root/smadex-creative/creative_intelligence`
**Verdict:** PASS — all targeted thresholds met; one minor non-blocking nit noted below.

---

## 1. Summary table

| Check | Status | Notes |
|---|---|---|
| 1. `scripts/train_all.py` re-trains cleanly | PASS | OOF Perf MAE 0.0251, status acc 0.80, macro F1 0.73 |
| 2. `pytest tests/ -v` (with and without `-m "not slow"`) | PASS | 23/23 passed in 4.10 s |
| 3. `python3 eval.py` | PASS | All thresholds met (table below) |
| 4. Gradio demo boots, returns HTTP 200, all 6 tabs render | PASS | All 6 tab labels present in HTML |
| 5. Manual pipeline integration (4 statuses × 4 methods + diversify) | PASS | See `tests/qa_pipeline_integration.py` |
| 6. Pipeline method signatures intact | PASS | `health_score`, `explain`, `find_similar`, `cluster_info` all present |
| 7. All disk artifacts present | PASS | All 12 expected files present, plus 8 extra `*_seed{1..4}.json` (5-seed bag) |

---

## 2. Headline metrics from `eval.py`

### Latency (30-creative sample, single-process, warm cache)

| Operation     | mean | p50 | p95  | max  | Threshold        | Status |
|---------------|-----:|----:|-----:|-----:|------------------|:------:|
| health_score  | 44.7 | 43.1| 53.4 | 58.8 | p95 < 100 ms     | PASS   |
| find_similar  | 12.1 | 11.9| 12.4 | 17.2 | p95 < 50 ms      | PASS   |
| cluster_info  |  0.1 |  0.1|  0.1 |  0.2 | (no spec)        | PASS   |
| explain       | 66.6 | 62.9| 77.1 |125.4 | p95 < 150 ms     | PASS   |
| Cold start    | 2.02 s | — | — | — | < 5 s            | PASS   |

### Action accuracy (n=200, sampled with seed=42)

| Status         | Strict match | Threshold | Status |
|----------------|-------------:|-----------|:------:|
| Overall strict | **200 / 200 = 100.0%** | ≥ 90% | PASS |
| top_performer  | 7 / 7   = 100.0% | ≥ 60% | PASS |
| stable         | 141 / 141 = 100.0% | (no spec) | PASS |
| fatigued       | 34 / 34 = 100.0% | ≥ 80% | PASS |
| underperformer | 18 / 18 = 100.0% | ≥ 80% | PASS |

Confusion is fully diagonal — every row is on its expected action.

### Recommender quality

- Mean same-status fraction in top-5: **58.93%** (random baseline ≈ 35.81%) — PASS (≥ 50%)
- For top_performer queries, mean (neighbor_perf − own_perf) = **−0.208**

### Cluster quality

- Clusters: 32; mean size 32.8; median 31
- **Vertical purity: mean 99.75%, min 96.97%** — PASS (≥ 99% mean)
- Noise points: 32 / 1080 (3.0%)

### Training metrics (from `/tmp/train.log`)

```
CV Perf MAE (OOF, GroupKFold):                                     0.0251 ± 0.0024
Status report (OOF, StratifiedGroupKFold + balanced+boost
                 + 5-seed bag + prior-bias):
  top_performer       precision 0.62  recall 0.67  F1 0.65   (n=46)
  stable              precision 0.87  recall 0.86  F1 0.87   (n=740)
  fatigued            precision 0.62  recall 0.63  F1 0.62   (n=199)
  underperformer      precision 0.78  recall 0.78  F1 0.78   (n=95)
  accuracy                                          0.80   (n=1080)
  macro avg                                         0.73
  weighted avg                                      0.80
Class-prior bias chosen:  [+0.90, −0.30, +0.10, −0.10]   (top, stable, fat, under)
Temperature scaling: T = 1.377;  ECE 0.0765 → 0.0333
```

All thresholds in the spec (Perf MAE ≤ 0.027, status acc ≥ 0.78, macro F1 ≥ 0.70) are exceeded.

---

## 3. Disk artifacts inventory

All 12 expected files present at `/root/smadex-creative/creative_intelligence/outputs/`:

```
outputs/embeddings/clip_embeddings.npz                  PRESENT
outputs/models/xgb_perf.json                            PRESENT
outputs/models/xgb_status.json                          PRESENT
outputs/models/tabular_meta.pkl                         PRESENT
outputs/models/temperature.pkl                          PRESENT
outputs/models/fatigue_clf.pkl                          PRESENT
outputs/models/fatigue_reg.pkl                          PRESENT
outputs/rubric/rubric_scores.parquet                    PRESENT
outputs/clusters/labels.parquet                         PRESENT
outputs/clusters/cluster_names.parquet                  PRESENT
outputs/knn/index.pkl                                   PRESENT
outputs/shap/background.npz                             PRESENT
```

Additional artifacts written by the 5-seed bag (expected, not in original list):
`xgb_perf_seed{1..4}.json`, `xgb_status_seed{1..4}.json`, plus
`outputs/clusters/umap_coords.npz` and `outputs/rubric/rubric_scores.jsonl`.

---

## 4. Pipeline integration spot-check

`tests/qa_pipeline_integration.py` (added by this QA run) cold-starts the pipeline
and exercises `health_score`, `explain`, `find_similar`, `cluster_info`, and
`find_similar(diversify=True)` against one creative per true status. All five
assertions pass for all four statuses (cids 500018 / 500000 / 500001 / 500006).

`diversify=True` returned 5 distinct creative_ids in every case and none of them
were the query creative. The diversified slate noticeably differs from the
non-diversified one (e.g. cid 500001: vanilla `[500864, 500828, 500688, 500972,
500401]` vs diversified `[500828, 500147, 500864, 500688, 500401]`).

---

## 5. Demo verification

- `python3 demo/app.py` booted in roughly 5 s and reached `Running on local URL: http://0.0.0.0:7860`.
- `curl http://localhost:7860/` → HTTP 200, 504,594-byte HTML response.
- Tab-label scan returned all 6 distinct labels: `Cluster Map`, `Explain`,
  `Health Score`, `Overview`, `Performance Explorer`, `Recommend`.
- Demo terminated cleanly via `pkill -f demo/app.py`.

Cosmetic warning (non-blocking): `gr.Blocks(theme=...)` is deprecated in
Gradio 6.0; theme should be passed to `launch()` instead. App still renders.

---

## 6. Regressions found

**None blocking.**

Minor / cosmetic:

1. **Top-level test files vs `tests/` directory** — `test_health_score.py` and
   `test_demo_surface.py` exist at the *repo root* alongside the proper
   `tests/test_health_score.py`. `pytest.ini` sets `testpaths = tests`, so the
   loose top-level files are silently ignored by `pytest tests/`. They aren't
   broken — they're just dead weight from earlier development. Recommend either
   deleting them or moving them under `tests/`.

2. **LightGBM "No further splits with positive gain" spam** during training —
   roughly 200 lines of warnings. Cosmetic only; the fitted model is fine.

3. **Sklearn warning during inference** — `LGBMClassifier` was fitted with
   feature names but is being called with a bare numpy array, producing
   `UserWarning: X does not have valid feature names...`. Predictions are
   correct (column order matches), but the warning is noisy. Consider passing a
   `pandas.DataFrame` with column names through the inference path, or fitting
   the LGBM model on numpy arrays.

4. **Gradio 6.0 deprecation** — see section 5.

5. **BOCPD claim** — the prior agent flagged a possible BOCPD issue. I refute
   that. The four `tests/test_bocpd.py` cases all pass cleanly:
   constant-series-no-CP, regime-shift-fires-CP, short-series-no-crash,
   probabilities-in-[0,1]. The implementation correctly short-circuits on
   `len < 4`, returns probabilities in the unit interval, fires on a clear
   regime shift with low threshold, and stays silent on a near-constant series
   with the production threshold of 0.4. No relaxed-threshold test was needed.

---

## 7. Claims vs reality

| Claim                                            | Measured                  | Match |
|--------------------------------------------------|---------------------------|:-----:|
| Status accuracy 0.79 → 0.80                      | 0.80                      | YES   |
| Macro F1 0.68 → 0.73                             | 0.73                      | YES   |
| top_performer F1 0.52 → 0.64                     | 0.65 (slightly better)    | YES   |
| OOF Perf MAE ≤ 0.027                             | 0.0251                    | YES   |
| 5-seed bagged XGBoost                            | base + 4 seed files (5)   | YES   |
| top_performer recall 14% → ~100% (action match)  | 100% on n=7 in eval sample | YES (small n) |
| Strict action match ≥ 90%                        | 100.0%                    | YES   |
| Recommender same-status ≥ 50%                    | 58.93%                    | YES   |
| Cluster vertical purity ≥ 99%                    | 99.75% mean, 96.97% min   | YES   |
| `diversify=True` returns 5 distinct ids          | 5 distinct, 4/4 statuses  | YES   |
| 23 pytest tests, all pass                        | 23 collected, 23 passed   | YES   |
| Demo has 6 tabs                                  | 6 tab labels in HTML      | YES   |

No discrepancies. One small caveat: the `top_performer` strict-match measurement
in the eval is over only n=7 creatives in the seed=42 sample, so the 100% number
is a small-sample observation rather than a population estimate. The OOF
classification F1 of 0.65 / recall 0.67 is the more conservative population
estimate.

---

## 8. Recommended next steps (low-priority)

1. Delete or relocate the stale top-level `test_health_score.py`,
   `test_demo_surface.py`, and `debug_inference.py`.
2. Suppress (or silence with `verbose=-1`) the LightGBM "no further splits"
   warnings during training so the log is readable.
3. Pass `feature_name=False` at LightGBM fit time, or always feed it a
   `DataFrame`, to silence the inference-time `UserWarning`.
4. Move the `theme` argument from `gr.Blocks(...)` to `demo.launch(theme=...)`
   to silence the Gradio 6.0 deprecation.
5. Consider adding a unit test that asserts `find_similar(diversify=True)`
   returns 5 distinct ids — not currently in `tests/test_pipeline_smoke.py`.

None of these block shipping. The five parallel agents' claims all hold up
under measurement.
