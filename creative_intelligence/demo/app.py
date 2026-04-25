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


def tab_explain(creative_id: int):
    if creative_id is None:
        return None, "<i>Pick a creative.</i>", "", None, "", ""
    creative_id = int(creative_id)
    e = P.explain(creative_id)
    h = e["health"]

    # Calibrated proxy: clamp predicted perf into [0,1] range (already 0..1ish)
    p_pred = float(min(1.0, max(0.0, e.get("perf_pred", h.get("predicted_perf", 0.5)))))
    pill = confidence_pill(p_pred)

    headline_md = f"### {e['headline']} &nbsp; {pill}\n{e['action_line']}"

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

    return asset_for(creative_id), headline_md, primary_md, fig, rubric_md, cf_md


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
- **Health Score** — 0–100 score with Scale / Continue / Pivot / Pause action for any creative.
- **Explain** — SHAP attributions, rubric callouts, and counterfactual experiments.
- **Recommend** — find the most similar top-performers (vertical-scoped or global, optional diversify).
- **Cluster Map** — UMAP of the full visual + behavioral genome, named clusters.
- **Performance Explorer** — slice the daily fact table by vertical / format / OS / country.
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
            demo.load(overview_metrics, [], [ov_cards, ov_status, ov_vert])

        # ---- Health Score tab ----
        with gr.Tab("Health Score"):
            gr.Markdown("Pick a creative to get its 0–100 Health Score and recommended action (Scale / Continue / Pivot / Pause).")
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
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    e_cid = gr.Dropdown(
                        choices=CIDS, value=CIDS[0], label="Creative ID",
                        info="Pick a creative to explain its predicted performance.",
                    )
                    e_img = gr.Image(label="Creative", height=320)
                with gr.Column(scale=2, min_width=420):
                    e_headline = gr.Markdown()
                    e_md = gr.Markdown()
                    e_chart = gr.Plot()
                    with gr.Accordion("Rubric callouts (extreme visual scores)", open=False):
                        e_rubric = gr.Markdown()
                    with gr.Accordion("Suggested counterfactual experiments", open=False):
                        e_cf = gr.Markdown()
            e_cid.change(tab_explain, [e_cid], [e_img, e_headline, e_md, e_chart, e_rubric, e_cf])
            demo.load(tab_explain, [e_cid], [e_img, e_headline, e_md, e_chart, e_rubric, e_cf])

        # ---- Recommend tab ----
        with gr.Tab("Recommend"):
            gr.Markdown("Find similar top performers — useful for cloning the visual style of a fatigued creative.")
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
