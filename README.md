# Smadex Creative Intelligence — Hackathon PoC

A full-stack creative-performance prediction system for mobile advertising creatives, built on the synthetic Smadex Creative Intelligence Challenge dataset (1,076 creatives × 36 advertisers × 6 verticals).

## Repo layout

```
.
├── README.md                   ← you are here
├── run.sh                      ← starts back+front in one command
├── .gitignore
│
├── models/                     ← all training & analysis code + saved artefacts
│   ├── notebooks/              5 executable notebooks (01_…05_)
│   ├── scripts/                build_clean_dataset.py, train_clean.py, etc.
│   ├── src/                    shared library (encoders, fatigue, embeddings)
│   ├── outputs/                splits, trained models, embeddings, rubric, kNN
│   ├── tests/                  pytest suite
│   ├── config.yaml
│   ├── requirements.txt
│   ├── PIPELINE.md             one-page reference
│   └── INSTRUCTIONS.md
│
├── back/                       ← FastAPI backend (12 endpoints)
│   └── main.py
│
├── front/                      ← React + TypeScript + Vite + Tailwind
│   ├── src/                    pages: Home / Stats / Explorer / Predict
│   ├── public/data/            embedded model predictions (works offline)
│   ├── package.json
│   └── README.md
│
├── assets/                     synthetic creative images
└── *.csv                       raw Smadex data (not regenerated)
```

## Quickstart — run back + front in one command

```bash
# 1. Install once (≈ 2 minutes)
pip install -r models/requirements.txt
(cd front && npm install)

# 2. Boot everything
./run.sh
```

This starts:

- **Backend**  — http://localhost:8000   (FastAPI, hot-reload)
- **Frontend** — http://localhost:5173   (Vite dev server, hot-reload)

The frontend is self-contained — it ships with precomputed predictions for all 1,076 creatives, so it works **with or without the backend running**.

## Reproduce the models from scratch

```bash
# (1) clean splits from raw CSVs (~5 sec)
python3 models/scripts/build_clean_dataset.py

# (2) train val-tuned baseline (~17 sec)
python3 models/scripts/train_clean.py

# (3) refit on train ∪ val for production deployment (~20 sec)
python3 models/scripts/train_clean.py --final

# (4) run the 5 notebooks in order (~3 minutes)
cd models/notebooks
for nb in 01_data_audit 02_data_analysis 03_dataset_balancing 04_models 05_evals; do
    jupyter nbconvert --to notebook --execute ${nb}.ipynb --output ${nb}.ipynb
done
```

End-to-end from raw CSVs → trained models → executed notebooks: **~4 minutes** on CPU.

## Headline test metrics (held-out n=216, no temperature scaling)

| Metric | Value |
|---|---|
| **macro-F1** | **0.677** |
| weighted-F1 | 0.781 |
| accuracy | 0.773 |
| AUC top_performer | 0.94 |
| AUC underperformer | 0.98 |
| Health Score → status Spearman | 0.45 |
| Fatigue 4-bucket macro-F1 | 0.43 |
| Per-class F1 | stable 0.84 · fatigued 0.63 · top_performer 0.60 · under 0.63 |

See `models/PIPELINE.md` for the full preprocessing breakdown, model paths, per-vertical numbers, and honest caveats.

## What's in each piece

**`models/`** — 6-model ensemble (XGBoost 5-seed bag, LightGBM, CatBoost, HistGBM, LogReg + soft-vote) trained on cleaned, leakage-free splits. Group-aware StratifiedGroupKFold by `campaign_id`. Notebooks 01–05 cover audit → analysis → balancing → models → evaluation.

**`back/`** — FastAPI app exposing 12 JSON endpoints over the trained pipeline (status prediction, health score, fatigue forecast, per-creative explanations, similarity search). Run via:
```bash
PYTHONPATH=models uvicorn back.main:app --reload
```

**`front/`** — React 18 + TypeScript + Vite + Tailwind + Framer Motion. Four pages: animated landing, stats dashboard, per-creative explorer, live prediction form. Static SPA, deployable to any CDN.

## Honest caveats

- The dataset is **synthetic**. A predict-by-vertical-prior baseline gets ~0.30 macro-F1 for free; our model adds +0.35 macro-F1 on top of that, mostly via early-life CTR aggregates.
- Top-performer F1 has a 95% CI of [0.31, 0.89] (n=11 in test) — the point estimate is directional.
- Pause/Pivot tier precision is 0.54 — should run as a *Watch* queue, not autonomous action.

## License

Synthetic data; all names, apps, assets, and metrics are fabricated. For learning, prototyping, and demos — not for production benchmarking.
