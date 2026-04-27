# Creative.AI — your AI creative companion

> A **proposal to improve mobile-ad campaigns**, submitted to the
> **Smadex Creative Intelligence Challenge** at **HackUPC 2026**.
>
> Drop an ad creative, get a Health Score, a structured analysis, an AI
> rebuild, and a live coach that lets you circle any element on the
> creative to ask follow-up questions.

Built on the Smadex Creative Intelligence Challenge dataset (1,076 mobile-ad
creatives × 36 advertisers × 6 verticals), the app chains **three
personalised models** to turn a screenshot into actionable recommendations.

## The three personalised models

| | Model | Role | Backed by |
|---|---|---|---|
| **1** | Soft-vote tabular ensemble | 4-class status + 0–100 Health Score + counterfactual lifts | XGBoost ×5 seeds + LightGBM + CatBoost + HistGBM + LogReg |
| **2** | Personalised VLM | Structured analysis JSON (strengths, weaknesses, fatigue reason, fixes) | SmolVLM-Instruct full fine-tune + SDFT |
| **3** | Image-edit | Ensemble-driven creative rebuild | Flux edit, rank-32 LoRA + reward-weighted DPO |

Plus, on the front-end:
- **14-day lifecycle forecast** — predicted CTR / impressions / ROAS curves from `train_lifecycle.py`.
- **Data-grounded color palette** — k-means over real top-performer images in your vertical.
- **Live coach** — circle any element on your creative and Maya tells you whether it works or how to fix it; chat with her about it after.

---

## Two quickstarts

### A. Run the app (frontend + backend)

For a working demo, no model training required.

```bash
# 1. Install once (~2 minutes)
pip install -r models/requirements.txt
(cd front && npm install)

# 2. Set the OpenRouter key — drives the live VLM analysis + image edit
echo "VITE_OPENROUTER_API_KEY=sk-or-..." > front/.env.local

# 3. Boot everything
./run.sh
```

- **Frontend** → http://localhost:5173
- **Backend**  → http://localhost:8000

The frontend is self-contained — it ships with precomputed predictions
for every creative, so it renders without the backend running. The
OpenRouter key is only needed for the live AI features (metadata
extraction, the analysis JSON, the AI image edit).

### B. Reproduce the models

For a full retrain from raw CSVs.

```bash
# Model 1 — Tabular ensemble (~25 s on CPU)
python3 models/scripts/build_clean_dataset.py
python3 models/scripts/train_clean.py --final

# Lifecycle curves + per-vertical palettes (front-end lookup tables)
PYTHONPATH=models python3 models/scripts/train_lifecycle.py
PYTHONPATH=models python3 models/scripts/build_palette_lookup.py
```

That's the CPU path — end-to-end Model 1 from raw CSVs to trained
ensemble in **~4 minutes**. For Models 2 and 3 (SmolVLM full FT + Flux
edit LoRA + DPO), see the workshop walkthrough at
[`models/README.md`](models/README.md), or jump straight to the docs at
[`models/docs/`](models/docs/).

---

## Repo layout

```
.
├── README.md                       ← you are here
├── run.sh                          ← starts back + front in one command
│
├── data/                           raw inputs (1,076 PNGs + 7 CSVs)
│   ├── assets/creative_<cid>.png
│   ├── creatives.csv  campaigns.csv  advertisers.csv
│   └── creative_summary.csv  campaign_summary.csv  creative_daily_country_os_stats.csv
│
├── models/                         training, analysis, saved artefacts
│   ├── README.md                   workshop-level navigation     ← start here for models
│   ├── docs/                       written deep-dives (01_…06_)
│   ├── notebooks/                  5 narrative notebooks (01_…05_)
│   ├── scripts/                    entry-point CLIs (one per pipeline phase)
│   ├── src/                        library code (data, embeddings, models, fatigue, …)
│   ├── tests/                      pytest suite
│   └── outputs/                    committed artefacts (splits · models · embeddings · …)
│
├── back/                           FastAPI backend
│   └── main.py
│
└── front/                          React 18 + TypeScript + Vite + Tailwind + Framer Motion
    ├── src/
    │   ├── pages/                  Home · Stats · Explorer · Predict
    │   ├── components/
    │   └── lib/                    openrouter.ts · predict.ts · data.ts
    ├── public/data/                predictions · metadata · final_metrics · lifecycle_curves · palettes
    ├── tests/                      Playwright smoke tests
    └── playwright.config.ts
```

## Frontend pages

- **Home** — Apple-style cinematic story with sticky scroll-driven scenes (problem → numbers → solution → product).
- **Stats** — 8-chapter "How we built it" walkthrough: problem, dataset, leakage-free splits, feature engineering, the three personalised models, results, key findings, honest caveats.
- **Explorer** — image-first gallery of all 1,076 creatives with filter chips and a click-to-open detail drawer.
- **Predict** — drop an ad, see Health Score + status + class probabilities + LLM analysis + suggested palette + layout/copy fixes + 14-day lifecycle forecast + counterfactual lifts. Then either **Generate improved version** (auto rebuild via Flux edit / Nano Banana) or **Edit & draw with live tips** (circle-to-ask coach mode with Maya).

## Headline metrics (held-out n=216, no temperature scaling)

| Metric | Value |
|---|---|
| **macro-F1** | **0.677** |
| weighted-F1 | 0.781 |
| accuracy | 0.773 |
| AUC top_performer | 0.94 |
| AUC underperformer | 0.98 |
| Health Score → status Spearman | 0.45 |
| Per-class F1 | stable 0.84 · fatigued 0.63 · top_performer 0.60 · under 0.63 |

See [`models/docs/06_evaluation.md`](models/docs/06_evaluation.md) for
the full breakdown (per-vertical numbers, calibration, latency, caveats).

## Tech stack

- **Models** — XGBoost · LightGBM · CatBoost · HistGradientBoosting · scikit-learn · SmolVLM · Flux edit · diffusers · peft
- **Backend** — FastAPI · uvicorn · pandas · pydantic
- **Frontend** — React 18 · TypeScript · Vite · Tailwind CSS · Framer Motion · React Router · lucide-react
- **Image-edit** — Gemini 2.5 Flash Image (Nano Banana) via OpenRouter for both training-time positives 
- **Analysis VLM** — Gemini 2.5 Flash Lite via OpenRouter at runtime; SmolVLM 2.2B as the swap-in for offline / on-device
- **Tests** — Playwright (6 specs covering hero, predict, stats, explorer, mobile)

## Honest caveats

- The dataset is **synthetic**. A predict-by-vertical-prior baseline gets ~0.30 macro-F1 for free; the ensemble adds +0.35 on top, mostly via early-life CTR aggregates.
- Top-performer F1 has a 95% bootstrap CI of [0.31, 0.89] (n=11 in test). Point estimate is directional.
- Pause / Pivot recommendation precision ≈ 0.54 — surface for human review, don't autopilot.
- Lifecycle curve XGBoost regressor has r² ≈ 0.07 on shape; we ship the **bucket-means** of real curves, not the regressor's per-sample prediction, because the bucket means are more honest.
- Cold-start: the 7-day early-life signal is the strongest feature. For brand-new creatives with zero impressions the model degrades to the visual-rubric + metadata baseline (~0.45 macro-F1).

## License

This project is released under the
[**Creative Commons Attribution-NonCommercial 4.0 International License**](https://creativecommons.org/licenses/by-nc/4.0/)
(**CC BY-NC 4.0**).

You are free to **share** and **adapt** the work — including the code, the
docs, and the trained-model artefacts — provided you:

- give **attribution** (credit the project + link the license + indicate changes), and
- use the material for **non-commercial** purposes only.

**Commercial use is not permitted.** This project is a HackUPC 2026
hackathon proposal intended for **educational, research, and demo
purposes**; the dataset is synthetic and the results are illustrative.
For commercial licensing, contact the authors.

The full legal text is in [`LICENSE`](LICENSE). SPDX identifier:
`CC-BY-NC-4.0`.

Synthetic data note: all names, apps, assets, and metrics in this repo
are fabricated. For learning, prototyping, and demos — not for production
benchmarking.

<sub><i>Note: where self-hosting our self-trained models (SmolVLM full
fine-tune for analysis, Flux edit LoRA + DPO for rebuilds) isn't feasible
— no GPU, no weights mounted, or running the static SPA standalone — the
app transparently falls back to third-party equivalents via OpenRouter
(Gemini 2.5 Flash Lite for the analysis JSON, Gemini 2.5 Flash Image /
Nano Banana for the rebuild). Same prompts, same output schema; only the
inference backend changes.</i></sub>
