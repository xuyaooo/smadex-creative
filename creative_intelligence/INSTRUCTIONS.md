# Smadex Creative Intelligence — Path-B Build

A fast, deterministic, no-runtime-LLM creative copilot. One cached "Creative
Genome" vector powers all 5 challenge paths (Performance Explorer, Fatigue
Detection, Explainability, Recommendation, Clustering). Cold start ≈ 1–2s,
per-query latency p95 < 100 ms.

---

## What's in the demo

Six tabs sharing one cached `CreativeIntelligencePipeline`:

1. **Overview** — portfolio dashboard (1,080 creatives, 36 advertisers, 180 campaigns)
2. **Health Score** — 0–100 score + action: Scale / Continue / Pivot / Pause
3. **Explain** — image + SHAP feature attributions + rubric callouts + counterfactuals + cached natural-language teacher annotation; optional regenerate-via-local-SmolVLM button
4. **Recommend** — vertical-scoped CLIP-NN, optional MMR diversification + perf re-rank
5. **Cluster Map** — UMAP 2-D projection of all 1,080 creatives, 32 named HDBSCAN clusters
6. **Performance Explorer** — slice the 192k-row daily fact table by vertical / format / OS / country

Each tab has a "How to read this view" and "Tech behind this" accordion explaining the model and methodology.

---

## Architecture at a glance

```
┌────────────────┐  ┌────────────────────┐  ┌─────────────────────┐  ┌────────────┐
│  Image (PNG)   │  │  Tabular metadata  │  │  Daily fact table   │  │  OpenRouter│
└───────┬────────┘  └─────────┬──────────┘  └──────────┬──────────┘  │   teacher  │
        │                     │                        │             │ (offline,  │
   CLIP ViT-B/32         77 OHE/LE +              first-7-day        │  cached)   │
   → 512d → PCA(32)      4 engineered ratios      aggregates → 21d   └─────┬──────┘
        │                     │                        │                   │
        └─────────────────────┴───────┬────────────────┴───────────────────┘
                                      ▼
                             ┌────────────────────┐
                             │ Genome Vector 145d │
                             │ + 15-d LLM rubric  │
                             └─────────┬──────────┘
                                       │
                  ┌────────────────────┼────────────────────┐
                  ▼                    ▼                    ▼
          ┌───────────────┐   ┌─────────────────┐   ┌──────────────────┐
          │ XGBoost x5    │   │ LightGBM        │   │ NN index (per    │
          │ (perf + 4-cls │   │ fatigue clf+reg │   │  vertical)       │
          │  status)      │   │ + BOCPD on CTR  │   │ + UMAP/HDBSCAN   │
          │ + temperature │   │                 │   │ + MMR re-rank    │
          │ + class bias  │   │                 │   │                  │
          └───────────────┘   └─────────────────┘   └──────────────────┘
```

All inference reads from cached `outputs/*` parquet/pkl/npz files. No live LLM.

---

## Setup

```bash
# 1. Python environment
conda create -n smadex-ci python=3.11 -y
conda activate smadex-ci
cd creative_intelligence
pip install -r requirements.txt
```

Optional (for actually running the offline teacher labeler / SmolVLM finetune):
```bash
export OPENROUTER_API_KEY=sk-or-...     # for rubric + annotation extraction
export HF_TOKEN=hf_...                   # avoids HF Hub rate limits
```

---

## Reproduce the demo end-to-end

The artifacts in `outputs/` are committed, so for a quick start you can skip
straight to step 5. The full reproduction:

### 1. Train the tabular + fatigue models

```bash
python3 scripts/train_all.py
```

What it does:
- Loads `creative_summary.csv` joined with `campaigns.csv` + `advertisers.csv`
- Builds 77 tabular features + 4 engineered ratios
- Pulls 32-dim PCA-reduced CLIP embeddings from `outputs/embeddings/clip_embeddings.npz`
- Computes 21-d early-life behavioral features from the first 7 days of `creative_daily_country_os_stats.csv`
- Concats the 15-d LLM rubric (skipped with a hint if not yet extracted)
- Runs 5-fold StratifiedGroupKFold (grouped by `campaign_id`) on a 5-seed XGBoost ensemble for status, GroupKFold for perf
- Searches a 4-D log-prob class bias on OOF for max macro-F1
- Fits a single-scalar temperature scaler (Guo ICML 2017)
- Trains LightGBM `(fatigue_clf, fatigue_reg)`
- Persists every model under `outputs/models/`

Expected output:
```
CV Perf MAE (OOF, GroupKFold): 0.0251 ± 0.0024
Status report (OOF, StratifiedGroupKFold + balanced+boost + 5-seed bag + prior-bias):
  top_performer  precision 0.62  recall 0.67  F1 0.65   (n=46)
  stable         precision 0.87  recall 0.86  F1 0.87   (n=740)
  fatigued       precision 0.62  recall 0.63  F1 0.62   (n=199)
  underperformer precision 0.78  recall 0.78  F1 0.78   (n=95)
  accuracy                                       0.80   (n=1080)
  macro avg                                      0.73
Temperature scaling: T = 1.377;  ECE 0.0765 → 0.0333
```

> **Honest-numbers caveat.** Macro F1 is OOF-tuned (the bias grid is fit on the
> same OOF predictions). A nested re-tune with hold-out gives macro F1 ≈ 0.69
> ± 0.03. Treat 0.73 as an upper bound, 0.69 as a lower bound for honest
> generalization. To produce a fully honest test set, see "Held-out evaluation"
> below.

### 2. Build clustering + KNN + SHAP background

```bash
python3 scripts/build_artifacts.py     # KNN (per-vertical) + UMAP + HDBSCAN + SHAP background
python3 scripts/name_clusters.py       # deterministic cluster names from modal metadata
```

Total ~30 s. Outputs land at:
```
outputs/knn/index.pkl           per-vertical NearestNeighbors indices
outputs/clusters/labels.parquet (cluster_id, umap_x, umap_y) per creative
outputs/clusters/cluster_names.parquet
outputs/shap/background.npz     stratified 32-row SHAP background
```

### 3. (Optional) Extract LLM rubric + teacher annotations via OpenRouter

```bash
export OPENROUTER_API_KEY=sk-or-...
python3 scripts/extract_rubric.py --model google/gemini-2.5-flash --workers 64           # ~$0.10, ~5 min
python3 scripts/label_with_openrouter.py --model google/gemini-2.5-flash --workers 64    # ~$1.50, ~5 min
python3 scripts/train_all.py        # re-train with the rubric
```

Rubric features go to `outputs/rubric/rubric_scores.parquet` (15 dims × 1080 rows, 0–10 anchored). Natural-language annotations go to `outputs/pseudo_labels/teacher_labels.jsonl`.

### 4. (Optional) LoRA-finetune SmolVLM for on-the-fly annotations

Only needed if you want a self-contained model that can annotate brand-new
uploaded creatives without calling OpenRouter. Tested on a single RTX 4090
(24 GB), takes ~15 min:

```bash
python3 scripts/finetune_smolvlm.py --epochs 1 --batch-size 1 --grad-accum 16
```

Training distills the cached gemini-2.5-flash JSON annotations into a
LoRA adapter (r=16, alpha=32, target=q/k/v/o_proj). Loss starts ~13.0 and
drops to ~0.49. The 39 MB adapter saves to `outputs/models/vlm_finetuned/`.
The demo's Explain tab shows a "Regenerate via local SmolVLM" button when
this directory is present.

### 5. Launch the demo

```bash
python3 demo/app.py
```

Open `http://localhost:7860`. Cold start ~1.2s on a warm cache; ~5s on first
boot if HuggingFace processor needs to download.

---

## Held-out evaluation (recommended for honest numbers)

The committed `qa_report.md` reports **OOF metrics with bias-tuning fit on the
same OOF data** — there's a documented ~3-pt optimism on macro F1. To get
honest numbers, run training with a 15% campaign-level held-out test set:

```bash
# (To be added: scripts/train_all.py --holdout-frac 0.15)
```

This carves a campaign-stratified 15% split *before* training, never used
during bias tuning or temperature fitting, and reports metrics on it. Expected
honest macro F1 ≈ 0.69, top_performer F1 ≈ 0.60, ECE ≈ 0.05.

---

## File map

```
creative_intelligence/
├── config.yaml
├── requirements.txt
├── INSTRUCTIONS.md (this file)
├── eval.py                       end-to-end metric + latency report
├── qa_report.md                  audit findings (held-out caveats inside)
│
├── demo/
│   └── app.py                    Gradio 6-tab UI
│
├── scripts/
│   ├── train_all.py              tabular + fatigue + calibration training
│   ├── build_artifacts.py        KNN + UMAP/HDBSCAN + SHAP backgrounds
│   ├── name_clusters.py          deterministic cluster names
│   ├── extract_rubric.py         OpenRouter rubric scorer (15 dims, cached)
│   ├── label_with_openrouter.py  OpenRouter teacher annotator (NL JSON, cached)
│   ├── finetune_smolvlm.py       LoRA SmolVLM-Instruct distillation
│   └── benchmark_rubric.py       ground-truth eval for picking a teacher model
│
├── src/
│   ├── data/{loader, feature_engineering, early_features, rubric_features,
│   │         time_series_features}.py
│   ├── embeddings/clip_encoder.py
│   ├── models/{tabular_model, fatigue_detector, recommender, vlm_model}.py
│   ├── calibration/temperature.py        single-scalar temperature scaling
│   ├── fatigue/{bocpd, health_score}.py  Bayesian CPD + 0-100 health blend
│   ├── inference/{pipeline, explainer, dpp_recommender, annotations,
│   │              vlm_inference}.py
│   └── training/{openrouter_rubric, openrouter_teacher,
│                 on_policy_distillation, teacher_labeling,
│                 continual_learning}.py
│
├── tests/                        pytest suite (24 cases)
└── outputs/                      precomputed artifacts (committed)
```

---

## SOTA / honest-state-of-pipeline

| Component | Verdict | Upgrade path |
|---|---|---|
| XGBoost 5-seed bag (tabular) | 🟢 ON-PAR | TabPFN v2 (Hollmann *Nature* 2025) for n<10k |
| BOCPD + LightGBM (fatigue) | 🟡 BEHIND | DeepSurv / RSF for time-to-event |
| Temperature scaling | 🟡 BEHIND | Beta calibration (Kull AISTATS 2017) for trees |
| **CLIP ViT-B/32 (visual)** | 🔴 **OUTDATED** | **SigLIP 2 base** — single biggest upgrade, propagates to recommender + clusters |
| NN + MMR (recommender) | 🟡 right-sized | Fast-greedy DPP-MAP (Chen NeurIPS 2018) above ~5k |
| UMAP + HDBSCAN | 🟢 ON-PAR | nothing urgent |
| SmolVLM v1 + 1-epoch LoRA | 🟡 BEHIND | SmolVLM2 + 3-5 epochs + MLP target_modules |

See `qa_report.md` for the full audit (statistical rigor, SOTA, production
readiness).

---

## Troubleshooting

**Demo's Cluster Map tab is empty** — by design, the 1080-point UMAP scatter
is the heaviest render in the demo so it doesn't fire at page load. Click
the "Render cluster map" button on that tab.

**`AutoModelForImageTextToText` import error** — upgrade transformers:
```bash
pip install -U transformers
```
The Path-B SmolVLM finetune needs transformers ≥ 4.45 (it was renamed from
`AutoModelForVision2Seq`).

**`umap-learn` / `hdbscan` import error** — these are real deps for
`build_artifacts.py`. Re-run `pip install -r requirements.txt`.

**SmolVLM finetune fails with "stack expects each tensor to be equal size"**
— SmolVLM produces variable image-patch counts per sample. Use
`--batch-size 1 --grad-accum N` instead of higher batch size (already the
default).

**OpenRouter rate-limit** — drop `--workers` to 16 or 32.
