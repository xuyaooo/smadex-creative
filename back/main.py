"""
FastAPI backend for the Creative Intelligence demo.

Layout (after the 2026-04 restructure):
    repo/
    ├── back/main.py        ← this file
    ├── models/             (notebooks, scripts, src, outputs, config)
    ├── front/              (React static SPA)
    └── *.csv               (raw Smadex data)

Run from the repo root:
    PYTHONPATH=models uvicorn back.main:app --host 0.0.0.0 --port 8000 --reload
"""
import sys
from pathlib import Path

# Resolve repo layout
ROOT = Path(__file__).resolve().parent.parent     # repo root
MODELS = ROOT / "models"                           # notebooks/scripts/src/outputs
sys.path.insert(0, str(MODELS))                    # so `from src.X import Y` works

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.inference.pipeline import CreativeIntelligencePipeline


# ---------------- pipeline (shared, loaded once) ----------------
print("Booting pipeline...")
PIPELINE = CreativeIntelligencePipeline(str(MODELS / "config.yaml"))
PIPELINE._ensure_models()
print("Pipeline ready.")

MASTER = PIPELINE._master_df
DAILY = PIPELINE._daily_df
ASSETS_DIR = (ROOT / "data" / "assets").resolve()


# ---------------- FastAPI app ----------------
app = FastAPI(
    title="Smadex Creative Intelligence API",
    version="1.0",
    description="JSON endpoints over the Path-B Creative Genome pipeline.",
)

# Allow Vue dev server (5173) and the static front-end (any port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Static assets ----------------
# Serve creative PNGs as /assets/creative_<cid>.png
if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# Serve the Vue front-end as /
FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")


# ---------------- Helpers ----------------
def _ensure_creative(cid: int):
    if cid not in MASTER["creative_id"].values:
        raise HTTPException(status_code=404, detail=f"Creative {cid} not found")


# ---------------- API: index / metadata ----------------
@app.get("/api/health")
def healthcheck():
    return {"status": "ok", "n_creatives": len(MASTER), "n_daily_rows": len(DAILY)}


@app.get("/api/overview")
def overview():
    """Portfolio metrics + status / vertical distributions for the Home page."""
    n_creatives = int(MASTER["creative_id"].nunique())
    n_advertisers = int(MASTER.get("advertiser_id", pd.Series()).nunique() or 0)
    n_campaigns = int(MASTER.get("campaign_id", pd.Series()).nunique() or 0)
    avg_perf = float(MASTER.get("perf_score", pd.Series([0])).mean())

    status_counts = (
        MASTER["creative_status"].value_counts()
        .reindex(["top_performer", "stable", "fatigued", "underperformer"]).fillna(0)
    )
    vertical_counts = MASTER["vertical"].value_counts()

    return {
        "totals": {
            "creatives": n_creatives,
            "advertisers": n_advertisers,
            "campaigns": n_campaigns,
            "avg_perf_score": round(avg_perf, 3),
        },
        "status_distribution": [
            {"status": k, "count": int(v)} for k, v in status_counts.items()
        ],
        "vertical_distribution": [
            {"vertical": k, "count": int(v)} for k, v in vertical_counts.items()
        ],
    }


@app.get("/api/creatives")
def list_creatives(vertical: str | None = None, status: str | None = None, limit: int = 100):
    """List creatives optionally filtered by vertical / status."""
    df = MASTER
    if vertical and vertical != "(all)":
        df = df[df["vertical"] == vertical]
    if status and status != "(all)":
        df = df[df["creative_status"] == status]
    rows = df.head(limit)[
        ["creative_id", "vertical", "format", "creative_status",
         "perf_score", "headline", "cta_text", "dominant_color"]
    ].to_dict(orient="records")
    return {"count": len(rows), "creatives": rows}


@app.get("/api/dimensions")
def dimensions():
    """All unique values for filter dropdowns."""
    return {
        "verticals": sorted(MASTER["vertical"].dropna().unique().tolist()),
        "formats": sorted(MASTER["format"].dropna().unique().tolist()),
        "statuses": ["top_performer", "stable", "fatigued", "underperformer"],
        "oses": sorted(DAILY["os"].dropna().unique().tolist()),
        "countries": sorted(DAILY["country"].dropna().unique().tolist()),
    }


# ---------------- API: per-creative analyses ----------------
@app.get("/api/creatives/{cid}/health")
def creative_health(cid: int):
    _ensure_creative(cid)
    return PIPELINE.health_score(cid)


@app.get("/api/creatives/{cid}/explain")
def creative_explain(cid: int):
    _ensure_creative(cid)
    return PIPELINE.explain(cid)


@app.get("/api/creatives/{cid}/similar")
def creative_similar(cid: int, k: int = 5, scope: str = "vertical", diversify: bool = False):
    _ensure_creative(cid)
    if scope not in ("vertical", "all"):
        raise HTTPException(status_code=400, detail="scope must be 'vertical' or 'all'")
    return {
        "creative_id": cid,
        "scope": scope,
        "diversify": diversify,
        "results": PIPELINE.find_similar(cid, k=k, scope=scope, diversify=diversify),
    }


@app.get("/api/creatives/{cid}/cluster")
def creative_cluster(cid: int):
    _ensure_creative(cid)
    return PIPELINE.cluster_info(cid)


@app.get("/api/creatives/{cid}/annotation")
def creative_annotation(cid: int):
    _ensure_creative(cid)
    annot = PIPELINE.annotation(cid)
    return annot or {"creative_id": cid, "annotation": None}


@app.get("/api/creatives/{cid}/timeseries")
def creative_timeseries(cid: int):
    """Daily impressions/clicks/CTR for plotting the lifecycle curve."""
    _ensure_creative(cid)
    ts = (
        DAILY[DAILY.creative_id == cid]
        .groupby("days_since_launch")
        .agg(impressions=("impressions", "sum"),
             clicks=("clicks", "sum"),
             spend_usd=("spend_usd", "sum"),
             revenue_usd=("revenue_usd", "sum"))
        .reset_index()
    )
    ts["ctr"] = (ts["clicks"] / ts["impressions"].replace(0, float("nan"))).fillna(0)
    return {"creative_id": cid, "rows": ts.to_dict(orient="records")}


# ---------------- API: clustering / explorer ----------------
@app.get("/api/clusters/map")
def cluster_map():
    """All creatives' UMAP coords + cluster IDs + names."""
    cdf = pd.read_parquet(ROOT / "outputs/clusters/labels.parquet").merge(
        MASTER[["creative_id", "vertical", "creative_status"]], on="creative_id"
    )
    names = (pd.read_parquet(ROOT / "outputs/clusters/cluster_names.parquet")
             .set_index("cluster_id")["name"].to_dict())
    cdf["cluster_name"] = cdf["cluster_id"].map(names).fillna("Outliers")
    return {
        "points": cdf.to_dict(orient="records"),
        "clusters": [
            {"cluster_id": int(cid), "name": str(name), "size": int(sub.size)}
            for cid, name in names.items()
            for sub in [cdf[cdf.cluster_id == cid]]
        ],
    }


@app.get("/api/ablations")
def ablations():
    """Full ablation history (the path from leaky baseline → current model)."""
    import json
    p = ROOT / "outputs/ablations.json"
    if not p.exists():
        return {"runs": []}
    return json.loads(p.read_text())


@app.get("/api/explorer")
def explorer(
    vertical: str = "(all)",
    format: str = "(all)",
    os_: str = "(all)",
    country: str = "(all)",
):
    """Slice the daily fact table on any combination of dimensions."""
    d = DAILY.copy()
    if vertical != "(all)":
        cids = MASTER[MASTER.vertical == vertical]["creative_id"]
        d = d[d.creative_id.isin(cids)]
    if format != "(all)":
        cids = MASTER[MASTER.format == format]["creative_id"]
        d = d[d.creative_id.isin(cids)]
    if os_ != "(all)":
        d = d[d.os == os_]
    if country != "(all)":
        d = d[d.country == country]

    if d.empty:
        return {"summary": None, "lifecycle": [], "n_rows": 0}

    by_day = (
        d.groupby("days_since_launch")
        .agg(impressions=("impressions", "sum"),
             clicks=("clicks", "sum"),
             spend=("spend_usd", "sum"),
             revenue=("revenue_usd", "sum"))
        .reset_index()
    )
    by_day["ctr"] = (by_day["clicks"] / by_day["impressions"].replace(0, float("nan"))).fillna(0)
    by_day["roas"] = (by_day["revenue"] / by_day["spend"].replace(0, float("nan"))).fillna(0)

    return {
        "filters": {"vertical": vertical, "format": format, "os": os_, "country": country},
        "n_rows": int(len(d)),
        "summary": {
            "impressions": int(d["impressions"].sum()),
            "clicks": int(d["clicks"].sum()),
            "spend_usd": round(float(d["spend_usd"].sum()), 2),
            "revenue_usd": round(float(d["revenue_usd"].sum()), 2),
            "overall_ctr": round(d["clicks"].sum() / max(d["impressions"].sum(), 1), 4),
            "overall_roas": round(d["revenue_usd"].sum() / max(d["spend_usd"].sum(), 1), 3),
        },
        "lifecycle": by_day.to_dict(orient="records"),
    }


# ---------------- API: VLM regenerate (optional, slow) ----------------
@app.post("/api/creatives/{cid}/regenerate-annotation")
def regenerate_annotation(cid: int):
    """Generate a fresh annotation via the local SmolVLM LoRA. ~9s per call."""
    _ensure_creative(cid)
    if not PIPELINE.vlm_available:
        raise HTTPException(
            status_code=503,
            detail="SmolVLM adapter not available locally. Run scripts/finetune_smolvlm.py first.",
        )
    out = PIPELINE.generate_annotation(cid)
    if out is None:
        raise HTTPException(status_code=500, detail="VLM generation failed")
    return out
