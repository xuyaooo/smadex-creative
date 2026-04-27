# 01 · Pipeline overview

> [↑ Index](../README.md) · [02 · Data pipeline →](02_data_pipeline.md)

This is the full reference for the modeling side of the project. Read this
first; the rest of the docs (`02_…06_`) drill into each phase with the
specific files where every step is implemented.

## What we ship

A 3-model chain that takes a mobile-ad creative + its launch metadata and
returns a verdict, an explanation, and a rebuild:

| | Model | Role | Backed by |
|---|---|---|---|
| **1** | Soft-vote tabular ensemble | 4-class status + Health Score (0–100) + counterfactual lifts | XGBoost ×5 seeds + LightGBM + CatBoost + HistGBM + LogReg |
| **2** | Personalized VLM | Structured analysis JSON (strengths, weaknesses, fatigue reason, fixes) | SmolVLM-Instruct full fine-tune + SDFT |
| **3** | Image-edit | Ensemble-driven creative rebuild | Flux edit, rank-32 LoRA + reward-weighted DPO |

Implementation entry points:
- Tabular ensemble — [`scripts/train_clean.py`](../scripts/train_clean.py)
- Personalized VLM — [`scripts/finetune_smolvlm_full.py`](../scripts/finetune_smolvlm_full.py)
- Flux edit — [`scripts/finetune_flux_edit.py`](../scripts/finetune_flux_edit.py)

## Pipeline at a glance

```
  ┌─────────────────────────┐
  │  raw CSVs (3 files)     │   creatives · campaigns · advertisers
  │  data/*.csv             │   creative_summary · creative_daily_country_os_stats
  └────────────┬────────────┘
               │  scripts/build_clean_dataset.py
               │   • drop 4 bad creatives + 13 leakage cols + low-signal cols
               │   • merge first-7d behavioral aggregates
               │   • KMeans-24 genome cluster
               │   • class-balanced weights × 1.7 boost on top_performer
               │   • StratifiedGroupKFold by campaign_id (70/15/15)
               ▼
  ┌─────────────────────────┐
  │  outputs/splits/        │   train 717 · val 143 · test 216 (no campaign overlap)
  └────────────┬────────────┘
               │
        ┌──────┴──────────────────────────────────────────────────┐
        ▼                                                          ▼
  scripts/train_clean.py                              scripts/precompute_embeddings.py
  (soft-vote ensemble + fatigue)                      scripts/build_artifacts.py
                                                      scripts/build_palette_lookup.py
        ▼                                                          ▼
  outputs/models/{clean,final}/                       outputs/embeddings/ + clusters/
  test macro-F1 = 0.677 (final)                       SigLIP-2 / CLIP + UMAP + HDBSCAN + kNN
                                                      per-vertical palettes (k-means)
                                                                   │
                                                                   ▼
                                              scripts/extract_rubric.py
                                              outputs/rubric/rubric_scores.parquet
                                                                   │
                                                                   ▼
                                              scripts/label_with_openrouter.py
                                              outputs/pseudo_labels/teacher_labels.jsonl
                                                                   │
                                                                   ▼
                                              scripts/finetune_smolvlm_full.py  (Model 2)
                                                                   │
                                              scripts/generate_flux_pairs.py +
                                              scripts/finetune_flux_edit.py     (Model 3)
```

## What lives where

| Folder | Purpose |
|---|---|
| [`scripts/`](../scripts/) | Entry-point CLIs for each pipeline phase |
| [`src/data/`](../src/data/) | Loaders, feature engineering, early-life features, rubric, time-series features |
| [`src/embeddings/`](../src/embeddings/) | CLIP / SigLIP encoder |
| [`src/models/`](../src/models/) | Tabular model, fatigue detector, recommender, VLM model |
| [`src/calibration/`](../src/calibration/) | Temperature scaling (kept around but not used in production) |
| [`src/fatigue/`](../src/fatigue/) | BOCPD changepoint detector + 0–100 health-score blend |
| [`src/inference/`](../src/inference/) | Pipeline, explainer, DPP recommender, annotations, VLM inference |
| [`src/training/`](../src/training/) | OpenRouter rubric + teacher, SDFT loop, continual learning |
| [`notebooks/`](../notebooks/) | 5 narrative notebooks (`01_…05_`) — audit · analysis · balancing · models · evals |
| [`outputs/`](../outputs/) | Committed artefacts: cleaned inputs, splits, models, embeddings, clusters, knn, rubric, SHAP background, pseudo-labels |
| [`tests/`](../tests/) | pytest suite (≈24 cases) |

## Reproduce end-to-end (CPU path)

The four-minute version, no GPU required:

```bash
cd models
pip install -r requirements.txt

# Tabular ensemble (Model 1) — ~25 s total
python3 scripts/build_clean_dataset.py
python3 scripts/train_clean.py --final

# Lifecycle curves + per-vertical palettes (front-end lookup tables)
PYTHONPATH=. python3 scripts/train_lifecycle.py
PYTHONPATH=. python3 scripts/build_palette_lookup.py

# Run the 5 narrative notebooks (~3 min total)
cd notebooks
for nb in 01_data_audit 02_data_analysis 03_dataset_balancing 04_models 05_evals; do
    jupyter nbconvert --to notebook --execute ${nb}.ipynb --output ${nb}.ipynb
done
```

Models 2 (VLM full FT) and 3 (Flux edit LoRA + DPO) need a single H100 80 GB
and OpenRouter credits — see [`05_personalized_vlm_and_image.md`](05_personalized_vlm_and_image.md).

## Architecture decisions

A few high-level calls that shaped the pipeline.

- **Why a 3-model chain, not one big multimodal model.** The tabular
  ensemble is the only piece tied to a real numeric label, so we keep
  it at the head: the VLM and the image-edit don't have to re-discover
  what "good" looks like — they apply a brief. We can also swap any
  one model without retraining the others.

- **Why the tabular ensemble drives the brief.** The status target is
  the dataset's ground truth. If a VLM graded creatives instead we'd
  be distilling its biases. Anchoring on the ensemble keeps the
  rebuild loop closed: Model 1 says what to fix, Model 3 applies it,
  Model 1 re-scores.

- **Why we accept a third-party fallback at runtime.** Self-hosting
  the trained SmolVLM + Flux LoRA needs a GPU and several GB of
  weights. The fallback (Gemini via OpenRouter) keeps the demo
  reachable without that — same prompts, same schema, only the
  inference backend changes.

## Where to read next

| | Doc | What's in it |
|---|---|---|
| **02** | [`02_data_pipeline.md`](02_data_pipeline.md) | Audit, cleaning, leakage handling, splits, sample weights |
| **03** | [`03_tabular_models.md`](03_tabular_models.md) | Soft-vote ensemble, fatigue, lifecycle, hyperparams |
| **04** | [`04_visual_intelligence.md`](04_visual_intelligence.md) | Embeddings, clusters, kNN, palettes, LLM rubric |
| **05** | [`05_personalized_vlm_and_image.md`](05_personalized_vlm_and_image.md) | SmolVLM full FT + SDFT, Flux edit LoRA + DPO |
| **06** | [`06_evaluation.md`](06_evaluation.md) | Headline metrics, per-class / per-vertical, calibration, latency, caveats |

---

[↑ Index](../README.md) · [02 · Data pipeline →](02_data_pipeline.md)
