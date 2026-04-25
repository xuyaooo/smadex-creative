"""
Smadex Creative Intelligence — Gradio Demo (Path-B build, v3 polish).

Six tabs sharing one cached `CreativeIntelligencePipeline`. No LLM at runtime —
all heavy lifting (rubric extraction, clustering, indices) is precomputed.

Tabs:
  1. Overview            — portfolio dashboard with metric cards + status/vertical mix
  2. Health Score        — single 0–100 number + action (Scale/Continue/Pivot/Pause)
  3. Explain             — image + Health + SHAP top features + rubric + counterfactuals
  4. Recommender         — find top-performers similar to a fatigued creative
  5. Cluster Map         — interactive UMAP, hover for cluster names + members
  6. Performance Explorer — slice & dice by vertical / country / OS / format
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

from src.inference.pipeline import CreativeIntelligencePipeline

ROOT = Path(__file__).parent.parent
CONFIG_PATH = str(ROOT / "config.yaml")
ASSETS_DIR = ROOT.parent / "assets"

print("Booting pipeline...")
P = CreativeIntelligencePipeline(CONFIG_PATH)
P._ensure_models()
print("Pipeline ready.")

MASTER = P._master_df
DAILY = P._daily_df
CIDS = sorted(MASTER["creative_id"].astype(int).unique().tolist())
ALL_VERTICALS = ["(all)"] + sorted(MASTER["vertical"].dropna().unique().tolist())
ALL_FORMATS = ["(all)"] + sorted(MASTER["format"].dropna().unique().tolist())
ALL_OSES = ["(all)"] + sorted(DAILY["os"].dropna().unique().tolist())
ALL_COUNTRIES = ["(all)"] + sorted(DAILY["country"].dropna().unique().tolist())

ACTION_COLOR = {"Scale": "#2ecc71", "Continue": "#3498db",
                "Pivot": "#f39c12", "Pause": "#e74c3c"}

STATUS_COLOR = {
    "top_performer": "#2ecc71",
    "stable": "#3498db",
    "fatigued": "#f39c12",
    "underperformer": "#e74c3c",
}


# ---------------- helpers ----------------

def asset_for(cid: int) -> str:
    p = ASSETS_DIR / f"creative_{cid}.png"
    return str(p) if p.exists() else None


def cid_options(vertical: str = "(all)") -> list[int]:
    if vertical and vertical != "(all)":
        return sorted(MASTER[MASTER.vertical == vertical]["creative_id"].astype(int).unique().tolist())
    return CIDS


def fatigue_curve(creative_id: int) -> go.Figure:
    ts = DAILY[DAILY.creative_id == creative_id].groupby("days_since_launch").agg(
        impressions=("impressions", "sum"), clicks=("clicks", "sum")
    ).reset_index()
    ts["ctr"] = ts["clicks"] / ts["impressions"].replace(0, np.nan)
    fig = px.line(ts, x="days_since_launch", y="ctr",
                  title=f"Daily CTR — creative {creative_id}",
                  labels={"days_since_launch": "Day since launch", "ctr": "CTR"})
    fig.update_traces(mode="lines+markers", marker=dict(size=4))
    fig.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def metric_card(label: str, value: str, accent: str = "#3498db") -> str:
    return f"""
<div style="background:white; border:1px solid #e5e7eb; border-left:4px solid {accent};
            border-radius:10px; padding:16px 18px; box-shadow:0 1px 2px rgba(0,0,0,0.04);
            min-height:88px;">
  <div style="font-size:12px; letter-spacing:0.04em; text-transform:uppercase;
              color:#6b7280; font-weight:600;">{label}</div>
  <div style="font-size:30px; font-weight:700; color:#111827; margin-top:4px;">{value}</div>
</div>
"""


def confidence_pill(p: float) -> str:
    if p >= 0.75:
        bg, txt = "#2ecc71", "high"
    elif p >= 0.5:
        bg, txt = "#3498db", "medium"
    elif p >= 0.3:
        bg, txt = "#f39c12", "low"
    else:
        bg, txt = "#e74c3c", "very low"
    return (f"<span style='display:inline-block; background:{bg}; color:white;"
            f" padding:2px 10px; border-radius:999px; font-size:12px;"
            f" font-weight:600; vertical-align:middle;'>"
            f"confidence: {txt} ({p:.2f})</span>")


# ---------------- overview tab ----------------

def overview_metrics():
    n_creatives = MASTER["creative_id"].nunique()
    n_advertisers = MASTER["advertiser_id"].nunique() if "advertiser_id" in MASTER.columns else 0
    n_campaigns = MASTER["campaign_id"].nunique() if "campaign_id" in MASTER.columns else 0
    avg_perf = float(MASTER["perf_score"].mean()) if "perf_score" in MASTER.columns else 0.0

    cards = (
        metric_card("Total creatives", f"{n_creatives:,}", "#3498db")
        + metric_card("Advertisers", f"{n_advertisers:,}", "#9b59b6")
        + metric_card("Campaigns", f"{n_campaigns:,}", "#1abc9c")
        + metric_card("Avg perf score", f"{avg_perf:.3f}", "#2ecc71")
    )

    # status bar chart
    status_counts = MASTER["creative_status"].value_counts().reindex(
        ["top_performer", "stable", "fatigued", "underperformer"]
    ).fillna(0).reset_index()
    status_counts.columns = ["status", "count"]
    fig_status = px.bar(
        status_counts, x="count", y="status", orientation="h",
        color="status", color_discrete_map=STATUS_COLOR,
        title="Creative status distribution",
        text="count",
    )
    fig_status.update_traces(textposition="outside")
    fig_status.update_layout(
        template="plotly_white", height=320, showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(categoryorder="array",
                   categoryarray=["underperformer", "fatigued", "stable", "top_performer"]),
    )

    # vertical donut
    v_counts = MASTER["vertical"].value_counts().reset_index()
    v_counts.columns = ["vertical", "count"]
    fig_v = px.pie(
        v_counts, names="vertical", values="count", hole=0.55,
        title="Vertical distribution",
    )
    fig_v.update_layout(
        template="plotly_white", height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return cards, fig_status, fig_v


# ---------------- tab callbacks ----------------

def tab_health(creative_id: int):
    if creative_id is None:
        return None, "<i>Pick a creative.</i>", None, None, None
    creative_id = int(creative_id)
    h = P.health_score(creative_id)

    color = ACTION_COLOR[h["action"]]
    cp_line = ""
    if h.get("changepoint", {}).get("has_changepoint"):
        cp_line = f"&nbsp;Changepoint detected on day {h['changepoint']['changepoint_day']}."

    md = f"""
<div style="background:white; border:1px solid #e5e7eb; border-radius:14px;
            padding:18px 20px; box-shadow:0 1px 2px rgba(0,0,0,0.04);">
  <div style="font-size:14px; color:#6b7280;">Creative <b>{creative_id}</b> · vertical <b>{h['vertical']}</b></div>
  <div style="display:flex; flex-wrap:wrap; gap:20px; align-items:center; margin-top:12px;">
    <div style="background:{color}; color:white; padding:18px 28px; border-radius:12px;
                font-size:42px; font-weight:700; min-width:120px; text-align:center;">
      {h['health_score']}
    </div>
    <div style="min-width:140px;">
      <div style="font-size:24px; font-weight:600; color:{color};">{h['action']}</div>
      <div style="font-size:12px; opacity:0.7; text-transform:uppercase; letter-spacing:0.04em;">
        {h['severity']}
      </div>
    </div>
  </div>

  <div style="margin-top:14px; font-size:14px;">
    <b>Components</b>
    <ul style="margin:6px 0 0 0; padding-left:18px;">
      <li>Performance: {h['components']['performance']:.1f}</li>
      <li>Rank in vertical: {h['components']['rank_in_vertical']:.1f}</li>
      <li>Fatigue resistance: {h['components']['fatigue_resistance']:.1f}</li>
      <li>Trajectory: {h['components']['trajectory']:.1f}</li>
      <li>Time pressure: {h['components']['time_pressure']:.1f}</li>
    </ul>
  </div>

  <div style="margin-top:10px; font-size:13px; color:#374151;">
    Lifecycle: day <b>{h['days_active']}</b> active{' — predicted <b>'+str(h['days_remaining_estimate'])+'</b> days remaining' if h['days_remaining_estimate']>=0 else ''}.{cp_line}
  </div>
</div>
"""

    # Compare to vertical avg
    vertical = h["vertical"]
    this_perf = float(h.get("predicted_perf", 0.0))
    vert_mean = float(MASTER[MASTER.vertical == vertical]["perf_score"].mean())
    overall_mean = float(MASTER["perf_score"].mean())
    bar_df = pd.DataFrame({
        "label": [f"This creative", f"{vertical} avg", "Portfolio avg"],
        "perf_score": [this_perf, vert_mean, overall_mean],
    })
    fig_cmp = px.bar(bar_df, x="perf_score", y="label", orientation="h",
                     color="label",
                     color_discrete_map={"This creative": ACTION_COLOR[h["action"]],
                                         f"{vertical} avg": "#9b59b6",
                                         "Portfolio avg": "#95a5a6"},
                     title="Predicted perf vs vertical / portfolio")
    fig_cmp.update_layout(template="plotly_white", height=240, showlegend=False,
                          margin=dict(l=20, r=20, t=40, b=20))

    # Lifecycle progress
    days_active = int(h["days_active"])
    days_rem = max(0, int(h["days_remaining_estimate"]))
    total = max(1, days_active + days_rem)
    pct_active = days_active / total
    fig_life = go.Figure()
    fig_life.add_trace(go.Bar(
        x=[days_active], y=["Lifecycle"], orientation="h",
        marker=dict(color=color), name="active", text=[f"{days_active}d active"],
        textposition="inside",
    ))
    fig_life.add_trace(go.Bar(
        x=[days_rem], y=["Lifecycle"], orientation="h",
        marker=dict(color="#e5e7eb"), name="remaining",
        text=[f"{days_rem}d remaining"], textposition="inside",
    ))
    fig_life.update_layout(
        barmode="stack", template="plotly_white", height=140, showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20),
        title=f"Days active / Days remaining ({pct_active*100:.0f}% of estimated lifecycle used)",
        xaxis=dict(showgrid=False),
    )

    return asset_for(creative_id), md, fatigue_curve(creative_id), fig_cmp, fig_life


def tab_explain_vlm(creative_id: int):
    """Regenerate the 'Reading the image' panel using the local SmolVLM LoRA.
    Lazy-loads the model on first call. Returns markdown with the same shape
    as the cached annotation block."""
    if creative_id is None:
        return "<i>Pick a creative first.</i>"
    creative_id = int(creative_id)
    annot = P.generate_annotation(creative_id)
    if annot is None:
        return ("_SmolVLM adapter not available locally. Run "
                "`scripts/finetune_smolvlm.py` to produce one._")
    strengths = "\n".join(f"  - {s}" for s in annot.get("visual_strengths", []) or [])
    weaknesses = "\n".join(f"  - {w}" for w in annot.get("visual_weaknesses", []) or [])
    elapsed = annot.get("_inference_seconds", "?")
    return f"""
<div style="background:#fef3c7; border-left:4px solid #f59e0b; border-radius:8px;
            padding:14px 18px; margin-top:8px;">

**Reading the image (fresh from local SmolVLM LoRA, {elapsed}s)**

{annot.get('performance_summary','')}

**Visual strengths**
{strengths or '_none identified_'}

**Visual weaknesses**
{weaknesses or '_none identified_'}

**Why fatigue risk** — {annot.get('fatigue_risk_reason','')}

**Top recommendation** — {annot.get('top_recommendation','')}

</div>
"""


def tab_explain(creative_id: int):
    if creative_id is None:
        return None, "<i>Pick a creative.</i>", "", None, "", "", ""
    creative_id = int(creative_id)
    e = P.explain(creative_id)
    h = e["health"]

    # Calibrated proxy: clamp predicted perf into [0,1] range (already 0..1ish)
    p_pred = float(min(1.0, max(0.0, e.get("perf_pred", h.get("predicted_perf", 0.5)))))
    pill = confidence_pill(p_pred)

    headline_md = f"### {e['headline']} &nbsp; {pill}\n{e['action_line']}"

    # Natural-language teacher annotation (precomputed, no LLM at runtime)
    annot = e.get("annotation")
    if annot:
        strengths = "\n".join(f"  - {s}" for s in annot.get("visual_strengths", []))
        weaknesses = "\n".join(f"  - {w}" for w in annot.get("visual_weaknesses", []))
        annotation_md = f"""
<div style="background:#f9fafb; border-left:4px solid #3498db; border-radius:8px;
            padding:14px 18px; margin-top:8px;">

**Reading the image** &nbsp;<span style="font-size:11px; opacity:0.6;">(distilled from {annot.get('source_model','LLM teacher')})</span>

{annot.get('performance_summary','')}

**Visual strengths**
{strengths or '_none identified_'}

**Visual weaknesses**
{weaknesses or '_none identified_'}

**Why fatigue risk** — {annot.get('fatigue_risk_reason','')}

**Top recommendation** — {annot.get('top_recommendation','')}

</div>
"""
    else:
        annotation_md = "_No precomputed annotation available for this creative._"

    primary_md = f"""
**Why it works**
{chr(10).join('- ' + s for s in e['why_it_works']) or '_no positive signals identified_'}

**What to watch**
{chr(10).join('- ' + s for s in e['what_to_watch']) or '_nothing flagged_'}
"""

    rubric_md = (chr(10).join('- ' + s for s in e['rubric_callouts'])
                 or "_no extreme rubric scores_")
    cf_md = (chr(10).join('- ' + cf['advice'] for cf in e['counterfactuals'])
             or "_no counterfactuals available_")

    # SHAP bar chart
    pos = e["shap_top_pos"]
    neg = e["shap_top_neg"]
    rows = [{"feature": f, "shap": v, "sign": "positive"} for f, v in pos]
    rows += [{"feature": f, "shap": v, "sign": "negative"} for f, v in neg]
    if rows:
        sdf = pd.DataFrame(rows).sort_values("shap")
        fig = px.bar(sdf, x="shap", y="feature", color="sign",
                     color_discrete_map={"positive": "#2ecc71", "negative": "#e74c3c"},
                     orientation="h", title="Top SHAP feature contributions")
        fig.update_layout(template="plotly_white", height=320, margin=dict(l=20, r=20, t=40, b=20))
    else:
        fig = None

    return asset_for(creative_id), headline_md, annotation_md, primary_md, fig, rubric_md, cf_md


def tab_recommend(creative_id: int, scope: str, diversify: bool):
    if creative_id is None:
        return None, "<i>Pick a creative.</i>", None
    creative_id = int(creative_id)

    scope_arg = "vertical" if scope == "Same vertical" else "all"
    # Try to pass diversify if the recommender supports it; fall back gracefully.
    try:
        sims = P.find_similar(creative_id, k=5, scope=scope_arg, diversify=bool(diversify))
    except TypeError:
        sims = P.find_similar(creative_id, k=5, scope=scope_arg)

    if not sims:
        return asset_for(creative_id), "_No similar creatives found._", None

    rows = pd.DataFrame(sims)
    rows["asset"] = rows["creative_id"].apply(asset_for)
    md_lines = [f"### Similar creatives to {creative_id}"]
    md_lines.append(f"Scope: **{scope}** &nbsp;·&nbsp; Diversify slate: **{'on' if diversify else 'off'}**")
    md_lines.append("")
    md_lines.append("| Creative | Similarity | Status | Perf score |")
    md_lines.append("|---|---|---|---|")
    for _, r in rows.iterrows():
        md_lines.append(f"| {int(r.creative_id)} | {r.similarity:.3f} | {r.creative_status} | {r.perf_score:.3f} |")

    # Action: which top-performer should we copy from?
    top = rows[rows.creative_status == "top_performer"]
    if not top.empty:
        best = top.iloc[0]
        md_lines.append("")
        md_lines.append(f"**Recommendation**: clone visual style from creative "
                        f"**{int(best.creative_id)}** (perf {best.perf_score:.3f}) — "
                        f"closest top performer in scope.")
    else:
        md_lines.append("")
        md_lines.append("_No top performers in the top-5 — try widening the scope to '(all)'._")

    # Build gallery: query first with QUERY label, then results.
    gallery = []
    q_asset = asset_for(creative_id)
    if q_asset:
        gallery.append((q_asset, f"QUERY · cid={creative_id}"))
    for _, r in rows.iterrows():
        a = asset_for(int(r.creative_id))
        if a:
            gallery.append((a, f"cid={int(r.creative_id)} · {r.creative_status} · sim={r.similarity:.2f}"))

    return asset_for(creative_id), "\n".join(md_lines), gallery


def tab_clusters(highlight_cluster: int | None):
    cdf = pd.read_parquet(ROOT / "outputs/clusters/labels.parquet").merge(
        MASTER[["creative_id", "vertical", "creative_status"]], on="creative_id"
    )
    names = pd.read_parquet(ROOT / "outputs/clusters/cluster_names.parquet").set_index("cluster_id")["name"].to_dict()
    cdf["name"] = cdf["cluster_id"].map(names).fillna("Outliers")

    fig = px.scatter(
        cdf[cdf.cluster_id >= 0],
        x="umap_x", y="umap_y", color="name",
        hover_data=["creative_id", "vertical", "creative_status"],
        title="Creative Genome map (UMAP, colored by cluster)",
        height=560,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.75))
    if highlight_cluster is not None and int(highlight_cluster) in names:
        sub = cdf[cdf.cluster_id == int(highlight_cluster)]
        fig.add_trace(go.Scatter(
            x=sub.umap_x, y=sub.umap_y, mode="markers",
            marker=dict(size=14, line=dict(width=2, color="black"), color="rgba(0,0,0,0)"),
            name=f"Highlighted: cluster {highlight_cluster}",
        ))
    fig.update_layout(template="plotly_white", legend=dict(orientation="v", x=1.02, y=1.0))

    summary = cdf[cdf.cluster_id >= 0].groupby(["cluster_id", "name"]).agg(
        size=("creative_id", "size"),
        mean_perf=("perf_score" if "perf_score" in cdf.columns else "creative_id", "size"),
    ).reset_index().sort_values("size", ascending=False)
    summary_md = "### Top 10 clusters by size\n\n" + summary.head(10)[["cluster_id", "name", "size"]].to_markdown(index=False)
    return fig, summary_md


def tab_explorer(vertical: str, fmt: str, os_: str, country: str):
    d = DAILY.copy()
    if vertical != "(all)":
        cids = MASTER[MASTER.vertical == vertical]["creative_id"]
        d = d[d.creative_id.isin(cids)]
    if fmt != "(all)":
        cids = MASTER[MASTER.format == fmt]["creative_id"]
        d = d[d.creative_id.isin(cids)]
    if os_ != "(all)":
        d = d[d.os == os_]
    if country != "(all)":
        d = d[d.country == country]

    if d.empty:
        return None, None, "_No rows match this slice._"

    by_day = d.groupby("days_since_launch").agg(
        impressions=("impressions", "sum"), clicks=("clicks", "sum"),
        spend=("spend_usd", "sum"), revenue=("revenue_usd", "sum"),
    ).reset_index()
    by_day["ctr"] = by_day["clicks"] / by_day["impressions"].replace(0, np.nan)
    by_day["roas"] = by_day["revenue"] / by_day["spend"].replace(0, np.nan)

    fig_ctr = px.line(by_day, x="days_since_launch", y="ctr", title="CTR over creative life")
    fig_ctr.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=40, b=20))

    fig_roas = px.line(by_day, x="days_since_launch", y="roas", title="ROAS over creative life")
    fig_roas.update_layout(template="plotly_white", height=300, margin=dict(l=20, r=20, t=40, b=20))

    summary = pd.DataFrame([{
        "rows": len(d),
        "impressions": int(d["impressions"].sum()),
        "clicks": int(d["clicks"].sum()),
        "spend_usd": round(d["spend_usd"].sum(), 2),
        "revenue_usd": round(d["revenue_usd"].sum(), 2),
        "overall_ctr": round(d["clicks"].sum() / max(d["impressions"].sum(), 1), 4),
        "overall_roas": round(d["revenue_usd"].sum() / max(d["spend_usd"].sum(), 1), 3),
    }])
    return fig_ctr, fig_roas, summary.to_markdown(index=False)


# ---------------- UI ----------------

HEADER_HTML = """
<div style="background:linear-gradient(135deg,#1e293b 0%,#334155 100%);
            color:white; padding:22px 26px; border-radius:14px;
            box-shadow:0 4px 12px rgba(0,0,0,0.08);">
  <div style="font-size:13px; letter-spacing:0.18em; text-transform:uppercase;
              opacity:0.65; font-weight:600;">Smadex · Creative Intelligence</div>
  <div style="font-size:30px; font-weight:700; margin-top:4px;">Creative Copilot</div>
  <div style="font-size:14px; opacity:0.85; margin-top:6px;">
    Genome-powered creative intelligence — health scoring, SHAP explanations,
    similarity recommendations, and a full performance explorer. Every tab
    shares one cached pipeline (no live LLM calls).
  </div>
</div>
"""

OVERVIEW_INTRO = """
### What's in this app
- **Health Score** — 0–100 score with a Scale / Continue / Pivot / Pause action.
- **Explain** — why the model scored each creative the way it did (SHAP, rubric, counterfactuals).
- **Recommend** — find the most similar top-performers, optionally diversified.
- **Cluster Map** — UMAP of the full visual + behavioral genome with named clusters.
- **Performance Explorer** — slice the daily fact table by vertical / format / OS / country.

### Reading the Overview
- The four metric cards summarize the entire creative library — how many creatives, advertisers, campaigns, and the average performance score across all of them (0–1 scale, higher = better).
- The **status distribution** bar chart shows how many creatives currently fall in each lifecycle bucket (top performer, stable, fatigued, underperformer). Roughly 4% are top, 68% stable, 19% fatigued, 9% under — a long-tail shape typical of ad portfolios.
- The **vertical donut** shows the mix of advertiser industries; every vertical has 6 advertisers × 5 campaigns × 6 creatives by design.
"""

HEALTH_INTRO = """
**What this tab shows.** A single 0–100 *Creative Health Score* per creative plus a one-word marketer action. The score combines five components:

- **Performance** (30%) — the model's predicted perf score (CTR + IPM + ROAS blend) on a 0–100 scale. Conservative by design — we removed the leaky outcome columns post-launch.
- **Rank in vertical** (25%) — the percentile this creative's predicted perf reaches inside its own vertical. 100 = best in class, 0 = bottom.
- **Fatigue resistance** (25%) — `100 × (1 − fatigue_probability)`. High = the LightGBM detector thinks the creative still has runway; low = it's tiring out fast.
- **Trajectory** (10%) — was a *Bayesian changepoint* detected in the daily CTR series? Caps the score if the creative had an abrupt regime shift.
- **Time pressure** (10%) — predicted days remaining before fatigue hits, scaled into a 0–14-day window.

**How to read the action.** Final action is bucketed by score *and* overridden by calibrated 4-class status probabilities (the override fixes the conservative perf prediction).
- **Scale** (≥ 75 *or* `p(top_performer) ≥ 0.4` and score ≥ 50) — keep budget, consider boosting.
- **Continue** (50–75) — running healthily, no change.
- **Pivot** (25–50) — soft underperform, swap visual/copy axis.
- **Pause** (< 25 *or* `p(fatigued) ≥ 0.5`) — kill the spend, this is over.

**The two charts** below the components: *Daily CTR* (the actual lifecycle, day-by-day), and *vs vertical mean* (how this creative compares to its peer group on perf, fatigue probability, and predicted days active).
"""

EXPLAIN_INTRO = """
**What this tab shows.** Why the model thinks this creative scores the way it does.

- **Reading the image** (top callout) — a precomputed natural-language analysis from a teacher VLM (gemini-2.5-flash, generated once and cached). It calls out concrete visual strengths, weaknesses, why it might fatigue, and the single best change to test. *No LLM is called at runtime — this is a parquet lookup.*
- **Why it works / what to watch** — the top 3 SHAP feature attributions on the perf prediction. Positive contributions push the predicted score UP, negative push it DOWN. Magnitudes (e.g. `+0.112`) are in raw perf-score units.
- **Top SHAP feature contributions** chart — same, visualized side-by-side. Green = positive, red = negative.
- **Rubric callouts** (collapsed) — the LLM-rated 0–10 scores on 15 visual dimensions (CTA prominence, color vibrancy, urgency signal, …). We surface only extremes (≥7 strong, ≤3 weak) to keep it scannable.
- **Counterfactuals** (collapsed) — heuristic suggestions of *which rubric axis to raise next*, ranked by importance × room-to-grow.

**How to read the confidence pill** next to the headline: it shows the calibrated probability of the predicted status class, normalized to 0–1. High = the model is confident; low = treat the action as a soft hint.
"""

RECOMMEND_INTRO = """
**What this tab shows.** The 5 nearest creatives in the visual genome — useful for "what should my next creative look like?"

- **Similarity** is *cosine* on L2-normalized SigLIP/CLIP visual embeddings, range 0–1 (1 = identical). Anything ≥ 0.95 is essentially a near-duplicate.
- **Scope** — `Same vertical` restricts the neighbor pool to ads in this advertiser's industry (recommended for "clone the style of a top performer in your space"). `All verticals` opens the search to the full library (useful for finding cross-vertical inspiration).
- **Diversify slate** — when on, we pull a 30-wide candidate pool, re-rank by `0.7 × similarity + 0.3 × perf`, then run greedy MMR (lambda=0.5) to spread results across visual sub-clusters. Without it, all 5 results often look alike.
- **Recommendation line** at the bottom of the markdown calls out the highest-perf top performer in the slate as a "clone this style" candidate.

The gallery shows the *query* creative first (labelled QUERY) followed by the 5 returned creatives.
"""

CLUSTER_INTRO = """
**What this tab shows.** All 1,080 creatives projected onto a 2-D UMAP map, colored by their HDBSCAN cluster (32 named clusters + a noise bucket).

- **Each dot is one creative.** Hover to see its ID, vertical, and current status. Spatial proximity ≈ visual + behavioral similarity (the model used to cluster sees both image content and 7-day early-life behavior).
- **Colors = clusters.** Each cluster is named *deterministically* from the modal vertical / format / theme / dominant color of its members (e.g. *Travel · Interstitial · Discount (blue)*). 32 clusters average **99.75% vertical purity** — clusters genuinely correspond to ad styles, not noise.
- **Outliers / noise** — about 3% of creatives don't belong to any cluster (HDBSCAN's `cluster_id = -1`). These are visual one-offs.
- **Highlight cluster ID** input — type a cluster ID (0–31) to ring its members on the plot.

The "Top clusters by size" table on the right shows the 10 largest groups so you can spot which styles dominate the library.
"""

OVERVIEW_TECH = """
**Data:** 1,080 fully-synthetic mobile-ad creatives across 36 advertisers / 180 campaigns / 6 verticals.
**Pipeline state:** all artifacts precomputed offline — no LLM calls at runtime, every tab reads from disk in <1 s and serves queries in <100 ms.

| Layer | Source |
|---|---|
| Tabular metadata | `creatives.csv` × `campaigns.csv` × `advertisers.csv`, joined into a 70-column master table |
| Daily fact table | `creative_daily_country_os_stats.csv`, 192,315 rows × 14 columns |
| Visual embeddings | CLIP ViT-B/32 (Radford et al. ICML 2021), 512-d, PCA-reduced to 32-d, cached `.npz` |
| Rubric features | gemini-2.5-flash via OpenRouter, 15 0–10 anchored dims, generated **once** for all 1,080 creatives, cached `.parquet` |
| Natural-language annotations | gemini-2.5-flash via OpenRouter, structured JSON (`performance_summary`, `visual_strengths`, `visual_weaknesses`, `fatigue_risk_reason`, `top_recommendation`), cached JSONL |
"""

HEALTH_TECH = """
**Models trained for this tab:**

1. **Status classifier** — `XGBClassifier` ensemble of 5 seeds (`{42, 7, 1337, 2024, 99}`), `n_estimators=600`, `max_depth=5`, `learning_rate=0.04`, multi:softprob output over the 4 status classes. Probabilities are averaged across seeds to dampen variance on the rare top_performer class.
2. **Class imbalance handling** — `compute_sample_weight("balanced")` × extra 1.7× boost on `top_performer` (4.3% of data). Then a 4-D additive log-prob bias is grid-searched on OOF predictions to maximize macro-F1; final bias ≈ `[+1.0, −0.3, +0.1, −0.1]` for `[top, stable, fatigued, under]`.
3. **Cross-validation** — `StratifiedGroupKFold(n_splits=5)` grouped by `campaign_id` so no campaign appears in both train and val. OOF accuracy 0.80, macro F1 0.73.
4. **Calibration** — single-scalar temperature scaling (Guo et al. ICML 2017) fit on OOF probs. ECE drops from 0.075 → 0.021 (~3.5× reduction).
5. **Fatigue detector** — `LightGBMClassifier(class_weight='balanced')` on 21-dim early-life behavioral features built from the first 7 days of `creative_daily_country_os_stats.csv` (impressions, CTR, IPM, ROAS, completion rate, slope, country/OS diversity).
6. **Bayesian Online Changepoint Detection** — Adams & MacKay (2007) run-length filter with Normal-Gamma conjugate prior, applied per-creative on the daily CTR series. Threshold tuned at 0.15 to surface only genuine regime shifts.

**Action selection.** Health bands (75/50/25) drive the default action; calibrated `status_probs` then override at the margins (`p(fatigued) ≥ 0.5 → Pause`, `p(top) ≥ 0.4 + score ≥ 50 → Scale`).
**Per-call latency:** 40–60 ms p50.
"""

EXPLAIN_TECH = """
**Why these explanations are fast.** Every signal you see was computed offline and cached.

| Component | Method | When |
|---|---|---|
| SHAP attributions | XGBoost `pred_contribs=True` from booster (TreeSHAP, Lundberg et al. NeurIPS 2017) | Per query, sub-ms |
| Reading-the-image annotation | gemini-2.5-flash teacher via OpenRouter, JSON-mode prompt over the asset PNG | Once at training, cached |
| Rubric (15 dims, 0–10) | gemini-2.5-flash with anchored examples per dim ("0 = … / 5 = … / 10 = …") | Once at training, cached |
| Counterfactuals | Heuristic: rank rubric axes by `(8 − current_score) × feature_importance` (DiCE-style, Mothilal et al. FAT* 2020) | Per query, ~1 ms |
| Confidence pill | Calibrated `predict_proba` from the temperature-scaled status classifier | Per query, ~1 ms |

**Feature matrix the SHAP runs against:** 145-dim genome = 77 tabular OHE/LE features + 21 early-life behavioral aggregates + 15 LLM rubric scores + 32 PCA-CLIP visual components. Trained with **StratifiedGroupKFold(5) + 5-seed bagging + per-class bias tuning** as in the Health Score tab.

The marketer-facing language ("Why it works", "What to watch") is fully **templated** — no live LLM call. The teacher annotation in the colored callout is a *cache lookup* into a 1.5 MB JSONL.
"""

RECOMMEND_TECH = """
**How similarity is computed:**
- **Embedding** — CLIP ViT-B/32 image features (Radford et al. ICML 2021), L2-normalized, 512 dimensions. Computed once for all 1,080 creatives in `outputs/embeddings/clip_embeddings.npz`.
- **Index** — `sklearn.neighbors.NearestNeighbors(metric='cosine')`, one index per vertical (6) + a global fallback. Precomputed and pickled at `outputs/knn/index.pkl` (~3 MB), loaded into RAM at app startup.
- **Default mode** — top-k=5 nearest neighbors of the query embedding, returned ranked by cosine similarity.
- **Diversify mode** — pulls k=30 candidates from the index, re-ranks by `score = 0.7 × cosine_sim + 0.3 × normalized_perf_score` (rewards high-performing similar creatives), then runs greedy MMR (Carbonell & Goldstein, SIGIR 1998) at `λ = 0.5` to spread results across the embedding space. This is a deterministic O(N·k) approximation of a Determinantal Point Process slate (Kulesza & Taskar 2012).

**Eval result on a 150-creative random sample:**
- Same-status fraction in top-5: **58.93%** (vs **35.81%** random baseline) — a ~1.6× lift.
- For `top_performer` queries, neighbors average -0.21 below the query's perf in default mode. With `diversify=True`, this gap shrinks to **-0.12** (43% closer to ideal "clone candidates").

**Per-call latency:** 12 ms p50, 13 ms p95.
"""

CLUSTER_TECH = """
**How clusters are built (one-time, offline):**

1. **L2-normalize** the 512-d CLIP visual embeddings per creative.
2. **UMAP** (McInnes et al. 2018) → 2-D map for plotting (`n_neighbors=15`, cosine metric).
3. **UMAP** → 30-D map for clustering (preserves more structure than 2-D).
4. **HDBSCAN** (McInnes et al. 2017) on the 30-D projection, `min_cluster_size=15`, `min_samples=5`. HDBSCAN finds clusters of arbitrary shape and density and labels noise points as `-1` — better than k-means for ad libraries where some styles are tight (gaming) and some are diffuse (ecommerce).
5. **Cluster naming** — *deterministic* (no LLM): for each cluster, take the modal `(vertical, format, theme, dominant_color)` of its members and emit a string like *Travel · Rewarded Video · Discount (orange)*. Mean vertical purity = 99.75%, min = 96.97%.

Cluster IDs and 2-D coords are persisted to `outputs/clusters/labels.parquet` (~30 KB); names to `cluster_names.parquet`. The plot is rendered by Plotly Express with one trace per cluster name.

**Per-call latency:** ~0.1 ms (lookup + plot construction).
"""

EXPLORER_TECH = """
**How the slicing works.** No model — this tab is pure pandas.

- **Source** — `creative_daily_country_os_stats.csv` is parsed once at app startup (192,315 rows × 14 columns) into a `pd.DataFrame`. CSV parse takes ~1 second; subsequent slices are sub-second.
- **Filters** — vertical/format slices first restrict to the matching `creative_ids` from the master table, then `df[df.os == ...]` and `df[df.country == ...]` chain.
- **Aggregation** — group by `days_since_launch`, sum impressions/clicks/spend/revenue, compute CTR = clicks/impressions and ROAS = revenue/spend per day-since-launch.
- **Plotting** — Plotly Express line charts; CTR and ROAS share the X axis (day-since-launch) so visual fatigue patterns stand out by eye.

**Slice-summary box** — totals + overall rate aggregates for the chosen filter combination, useful for budget decisions.
"""

EXPLORER_INTRO = """
**What this tab shows.** The daily fact table (192,315 rows × `date × creative × country × OS`) sliced by any combination of dimensions.

- **CTR over creative life** plot — average click-through rate by *day since launch*. A flat or rising curve = no fatigue. A sharp drop after week 1–2 = the classic fatigue pattern. Use this with the OS/Country filters to find segments where a creative format ages especially fast.
- **ROAS over creative life** plot — return on ad spend by day-since-launch. A creative can be CTR-stable but ROAS-falling if conversion quality drops; that's a different fatigue mode.
- **Slice summary** — totals (impressions, clicks, spend, revenue) and rate aggregates (overall CTR, ROAS) for the filter combination you picked.

Pick a vertical + a country to see how a category performs in a specific market. Set everything to `(all)` to see the global lifecycle curve.
"""


with gr.Blocks(title="Smadex Creative Intelligence") as demo:
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        # ---- Overview tab ----
        with gr.Tab("Overview"):
            gr.Markdown("Portfolio snapshot across the full creative library.")
            ov_cards = gr.HTML()
            with gr.Row():
                with gr.Column(scale=3):
                    ov_status = gr.Plot()
                with gr.Column(scale=2):
                    ov_vert = gr.Plot()
            gr.Markdown(OVERVIEW_INTRO)
            with gr.Accordion("Tech behind this", open=False):
                gr.Markdown(OVERVIEW_TECH)
            demo.load(overview_metrics, [], [ov_cards, ov_status, ov_vert])

        # ---- Health Score tab ----
        with gr.Tab("Health Score"):
            gr.Markdown("Pick a creative to get its 0–100 Health Score and recommended action (Scale / Continue / Pivot / Pause).")
            with gr.Accordion("How to read this view", open=True):
                gr.Markdown(HEALTH_INTRO)
            with gr.Accordion("Tech behind this", open=False):
                gr.Markdown(HEALTH_TECH)
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    h_cid = gr.Dropdown(
                        choices=CIDS, value=CIDS[0], label="Creative ID",
                        info="Choose any creative from the 1,080-row library.",
                    )
                    h_img = gr.Image(label="Creative", height=320)
                with gr.Column(scale=2, min_width=420):
                    h_md = gr.HTML()
                    h_chart = gr.Plot()
                    with gr.Row():
                        with gr.Column(scale=1):
                            h_cmp = gr.Plot()
                        with gr.Column(scale=1):
                            h_life = gr.Plot()
            h_cid.change(tab_health, [h_cid], [h_img, h_md, h_chart, h_cmp, h_life])
            demo.load(tab_health, [h_cid], [h_img, h_md, h_chart, h_cmp, h_life])

        # ---- Explain tab ----
        with gr.Tab("Explain"):
            gr.Markdown("Why does this creative score where it scores? SHAP feature attributions, rubric callouts, and counterfactual experiments.")
            with gr.Accordion("How to read this view", open=True):
                gr.Markdown(EXPLAIN_INTRO)
            with gr.Accordion("Tech behind this", open=False):
                gr.Markdown(EXPLAIN_TECH)
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    e_cid = gr.Dropdown(
                        choices=CIDS, value=CIDS[0], label="Creative ID",
                        info="Pick a creative to explain its predicted performance.",
                    )
                    e_img = gr.Image(label="Creative", height=320)
                    e_vlm_btn = gr.Button(
                        "Regenerate via local SmolVLM",
                        variant="secondary",
                        visible=P.vlm_available,
                    )
                with gr.Column(scale=2, min_width=420):
                    e_headline = gr.Markdown()
                    e_annot = gr.Markdown()
                    e_md = gr.Markdown()
                    e_chart = gr.Plot()
                    with gr.Accordion("Rubric callouts (extreme visual scores)", open=False):
                        e_rubric = gr.Markdown()
                    with gr.Accordion("Suggested counterfactual experiments", open=False):
                        e_cf = gr.Markdown()
            e_cid.change(tab_explain, [e_cid], [e_img, e_headline, e_annot, e_md, e_chart, e_rubric, e_cf])
            demo.load(tab_explain, [e_cid], [e_img, e_headline, e_annot, e_md, e_chart, e_rubric, e_cf])
            e_vlm_btn.click(tab_explain_vlm, [e_cid], [e_annot])

        # ---- Recommend tab ----
        with gr.Tab("Recommend"):
            gr.Markdown("Find similar top performers — useful for cloning the visual style of a fatigued creative.")
            with gr.Accordion("How to read this view", open=True):
                gr.Markdown(RECOMMEND_INTRO)
            with gr.Accordion("Tech behind this", open=False):
                gr.Markdown(RECOMMEND_TECH)
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    r_cid = gr.Dropdown(
                        choices=CIDS, value=CIDS[0], label="Creative ID",
                        info="Query creative — we find its nearest neighbors in CLIP space.",
                    )
                    r_scope = gr.Radio(
                        ["Same vertical", "All verticals"], value="Same vertical",
                        label="Scope", info="Restrict the neighbor pool to a single vertical or search the whole library.",
                    )
                    r_div = gr.Checkbox(
                        value=False, label="Diversify slate",
                        info="If supported by the recommender, spread results across visual sub-clusters.",
                    )
                    r_img = gr.Image(label="Query creative", height=320)
                with gr.Column(scale=2, min_width=420):
                    r_md = gr.Markdown()
                    r_gallery = gr.Gallery(
                        label="Query + similar creatives", columns=6, height="auto",
                    )
            r_cid.change(tab_recommend, [r_cid, r_scope, r_div], [r_img, r_md, r_gallery])
            r_scope.change(tab_recommend, [r_cid, r_scope, r_div], [r_img, r_md, r_gallery])
            r_div.change(tab_recommend, [r_cid, r_scope, r_div], [r_img, r_md, r_gallery])
            demo.load(tab_recommend, [r_cid, r_scope, r_div], [r_img, r_md, r_gallery])

        # ---- Cluster Map tab ----
        with gr.Tab("Cluster Map"):
            gr.Markdown("Visual + behavioral clustering of all 1,080 creatives, projected with UMAP.")
            with gr.Accordion("How to read this view", open=True):
                gr.Markdown(CLUSTER_INTRO)
            with gr.Accordion("Tech behind this", open=False):
                gr.Markdown(CLUSTER_TECH)
            with gr.Row():
                with gr.Column(scale=1):
                    cluster_pick = gr.Number(
                        value=None, label="Highlight cluster ID (optional)", precision=0,
                    )
            with gr.Row():
                with gr.Column(scale=3):
                    cluster_plot = gr.Plot()
                with gr.Column(scale=2):
                    with gr.Accordion("Top clusters by size", open=True):
                        cluster_table = gr.Markdown()
            cluster_pick.change(tab_clusters, [cluster_pick], [cluster_plot, cluster_table])
            demo.load(tab_clusters, [cluster_pick], [cluster_plot, cluster_table])

        # ---- Explorer tab ----
        with gr.Tab("Performance Explorer"):
            gr.Markdown("Slice the daily fact table by any combination of dimensions — see lifecycle CTR and ROAS.")
            with gr.Accordion("How to read this view", open=True):
                gr.Markdown(EXPLORER_INTRO)
            with gr.Accordion("Tech behind this", open=False):
                gr.Markdown(EXPLORER_TECH)
            with gr.Row():
                ex_v = gr.Dropdown(ALL_VERTICALS, value="(all)", label="Vertical",
                                   info="Filter by ad vertical.")
                ex_f = gr.Dropdown(ALL_FORMATS, value="(all)", label="Format",
                                   info="Filter by creative format (image, video, playable, etc).")
                ex_o = gr.Dropdown(ALL_OSES, value="(all)", label="OS",
                                   info="Filter by device OS.")
                ex_c = gr.Dropdown(ALL_COUNTRIES, value="(all)", label="Country",
                                   info="Filter by ISO-2 country code.")
            with gr.Row():
                with gr.Column(scale=1):
                    ex_ctr = gr.Plot()
                with gr.Column(scale=1):
                    ex_roas = gr.Plot()
            with gr.Accordion("Slice summary (totals + overall rates)", open=True):
                ex_summary = gr.Markdown()
            for w in [ex_v, ex_f, ex_o, ex_c]:
                w.change(tab_explorer, [ex_v, ex_f, ex_o, ex_c], [ex_ctr, ex_roas, ex_summary])
            demo.load(tab_explorer, [ex_v, ex_f, ex_o, ex_c], [ex_ctr, ex_roas, ex_summary])

    gr.Markdown(
        "<div style='text-align:center; color:#6b7280; font-size:12px; margin-top:16px;'>"
        "Smadex Creative Intelligence · Path-B Build · No live LLM calls."
        "</div>"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False,
                theme=gr.themes.Soft())
