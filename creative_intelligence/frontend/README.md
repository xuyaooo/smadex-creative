# Creative Intelligence Frontend

React 18 + TypeScript + Vite + Tailwind + Framer Motion. Static SPA — runs entirely client-side using a precomputed prediction bundle, so it works **without a backend running**.

## Stack

- **React 18** + TypeScript
- **Vite 5** for build / dev server
- **Tailwind CSS** for styling (matches the Smadex template's design system)
- **Framer Motion** for animations
- **Lucide React** for icons
- **React Router 6** for navigation

## Pages

| Route | What it shows |
|---|---|
| `/` | Landing — hero, animated blobs, headline metrics, feature grid |
| `/stats` | Dashboard — confusion matrix, per-class distribution, per-vertical accuracy, per-model val F1 |
| `/explorer` | Per-creative grid — filter by split / vertical / status / action; click for SHAP-style detail panel |
| `/predict` | Live form — pick vertical/format/color/flags + early CTR sliders → Health Score, predicted status, "what to change" suggestions |

## Data sources (embedded JSON)

Located under `public/data/` — produced by an export script that runs the production ensemble against all 1,076 creatives.

- `predictions.json` (686 KB) — every creative with predicted status, class probas, health score, fatigue prediction
- `metadata.json` — vocabularies for the form (verticals, formats, colors, themes, hooks, tones)
- `final_metrics.json` — production model test metrics + confusion matrix
- `eval_report.json` — full evaluation output (AUCs, per-vertical, Spearman, etc.)

## Develop

```bash
cd frontend
npm install
npm run dev   # → http://localhost:5173
```

## Build

```bash
npm run build         # → dist/
npm run preview       # serve dist/ locally
```

The `dist/` folder is fully static — drop it on Vercel/Netlify/Cloudflare Pages and it works without any backend.

## How predictions work in the live form

Since the page is static, `/predict` doesn't run XGBoost/CatBoost/LightGBM in-browser (those models are 11+ MB and don't ship with browser-friendly inference). Instead it does a **k=1 nearest-neighbor lookup** in the 1,076 precomputed predictions, weighted by feature similarity. With 6 verticals × 4 formats × 7 colors × 4 hook types × 4 themes × 5 tones, every plausible input combination has a close neighbor.

For exact model inference on novel creatives, point the app at the FastAPI backend at `localhost:8000` (deferred — see `backend/main.py`).
