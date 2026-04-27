# models/ — Creative Intelligence pipeline

This is the modeling side of the project: data prep, three personalised
models, and the evals behind every number on the front-end.

If you came here from the workshop or are reading the repo for the first
time, the quickest tour is:

> **1.** Read [`docs/01_pipeline_overview.md`](docs/01_pipeline_overview.md) for the full picture.
> **2.** Open [`notebooks/01_data_audit.ipynb`](notebooks/01_data_audit.ipynb) to see the dataset bug-hunt that motivated every preprocessing decision.
> **3.** Step through `02_…05_` notebooks in order — they are the narrative version of the pipeline, with plots.
> **4.** Drill into [`docs/02_…06_`](docs) for the corresponding written deep-dives, each pointing back to the exact `scripts/` and `src/` files where things are implemented.

## Quickstart (CPU, ~4 minutes)

The fastest path to a trained Model 1 + lifecycle curves + palettes —
everything the front-end needs:

```bash
cd models
pip install -r requirements.txt

# Build leakage-free splits (~5 s)
python3 scripts/build_clean_dataset.py

# Train the 5-model soft-vote ensemble — production refit (~25 s)
python3 scripts/train_clean.py --final

# Lifecycle curves + per-vertical palettes (front-end lookup tables)
PYTHONPATH=. python3 scripts/train_lifecycle.py
PYTHONPATH=. python3 scripts/build_palette_lookup.py
```

Outputs land under `outputs/`. End-to-end test macro-F1 = **0.677** —
see [`docs/06_evaluation.md`](docs/06_evaluation.md) for the full eval.

For Models 2 (SmolVLM full FT) and 3 (Flux edit LoRA + DPO), see
[`docs/05_personalized_vlm_and_image.md`](docs/05_personalized_vlm_and_image.md).
Both need a single H100 80 GB and OpenRouter credits.

## Workshop navigation (read in this order)

| | Read | Then run | Why |
|---|---|---|---|
| **Step 1** | [`docs/01_pipeline_overview.md`](docs/01_pipeline_overview.md) | — | Big picture, file map, where to look next |
| **Step 2** | [`docs/02_data_pipeline.md`](docs/02_data_pipeline.md) | [`notebooks/01_data_audit.ipynb`](notebooks/01_data_audit.ipynb) → [`notebooks/03_dataset_balancing.ipynb`](notebooks/03_dataset_balancing.ipynb) | Why every preprocessing decision was made |
| **Step 3** | [`docs/03_tabular_models.md`](docs/03_tabular_models.md) | [`notebooks/04_models.ipynb`](notebooks/04_models.ipynb) | The 5-model soft-vote ensemble, fatigue, lifecycle |
| **Step 4** | [`docs/04_visual_intelligence.md`](docs/04_visual_intelligence.md) | [`notebooks/02_data_analysis.ipynb`](notebooks/02_data_analysis.ipynb) | Embeddings, clusters, palettes, rubric |
| **Step 5** | [`docs/05_personalized_vlm_and_image.md`](docs/05_personalized_vlm_and_image.md) | (GPU only) | SmolVLM full FT + Flux LoRA + DPO |
| **Step 6** | [`docs/06_evaluation.md`](docs/06_evaluation.md) | [`notebooks/05_evals.ipynb`](notebooks/05_evals.ipynb) | Headline metrics, calibration, latency, caveats |

## Folder map

```
models/
├── README.md                       ← you are here
├── docs/                           ordered written reference (01_…06_)
│
├── notebooks/                      narrative pipeline, run in order
│   ├── 01_data_audit.ipynb            bug-hunt the raw data
│   ├── 02_data_analysis.ipynb         pattern discovery (archetypes, genome clusters)
│   ├── 03_dataset_balancing.ipynb     splitting + balancing walkthrough
│   ├── 04_models.ipynb                train + benchmark + TreeSHAP + counterfactuals
│   └── 05_evals.ipynb                 per-class CIs · per-vertical · ROC · reliability
│
├── scripts/                        entry-point CLIs (one file per phase)
│   ├── build_clean_dataset.py         splits + sample weights
│   ├── train_clean.py                 5-model soft-vote ensemble (Model 1)
│   ├── train_lifecycle.py             14-day lifecycle curves + archetype
│   ├── precompute_embeddings.py       SigLIP-2 / CLIP encoder
│   ├── build_artifacts.py             UMAP + HDBSCAN + per-vertical kNN
│   ├── name_clusters.py               deterministic cluster names
│   ├── build_palette_lookup.py        per-vertical hex palettes (k-means)
│   ├── extract_rubric.py              15-dim LLM rubric (optional features)
│   ├── benchmark_rubric.py            evaluate teacher-model candidates
│   ├── label_with_openrouter.py       teacher pseudo-labels (Model 2 input)
│   ├── finetune_smolvlm.py            LoRA fine-tune (legacy)
│   ├── finetune_smolvlm_full.py       FULL fine-tune SmolVLM (Model 2)        ★
│   ├── generate_flux_pairs.py         Nano Banana → (src, brief, target)      ★
│   ├── finetune_flux_edit.py          LoRA + reward-weighted DPO (Model 3)    ★
│   └── …                              (additional helpers — see scripts/ listing)
│
├── src/                            library code (imported by scripts + notebooks)
│   ├── data/                          loaders, feature engineering, early-life features, rubric
│   ├── embeddings/                    CLIP / SigLIP encoder
│   ├── models/                        tabular model, fatigue detector, recommender, VLM model
│   ├── calibration/                   temperature scaling
│   ├── fatigue/                       BOCPD changepoint + 0–100 health-score blend
│   ├── inference/                     pipeline, explainer, DPP recommender, VLM inference
│   ├── training/                      OpenRouter rubric + teacher, SDFT, continual learning
│   └── evaluation/                    metric helpers
│
├── tests/                          pytest suite (≈ 24 cases)
├── outputs/                        committed artefacts (splits · models · embeddings · clusters · rubric · pseudo-labels · flux_pairs)
├── figures/                        static figures used in docs / front-end
├── config.yaml                     data + model paths
├── conftest.py · pytest.ini        test config
├── eval.py                         end-to-end metric + latency report
└── requirements.txt
```

★ = added in this iteration of the project.

## Reproduce the published numbers

The CI command — runs the full Model 1 path and prints metric + latency
deltas vs. the thresholds in [`docs/06_evaluation.md`](docs/06_evaluation.md):

```bash
python3 scripts/build_clean_dataset.py
python3 scripts/train_clean.py --final
PYTHONPATH=. python3 eval.py
pytest tests/ -v
```

Wall-clock: ~30 s training + ~10 s eval + ~5 s tests on CPU.
